//! Advanced overhead comparison benchmarks for torsh-profiler
//!
//! This benchmark suite compares the performance impact of different profiling modes
//! to help users understand the trade-offs between profiling detail and overhead.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::thread;
use std::time::Duration;
use torsh_profiler::{
    clear_global_events, export_global_events, profile_scope, set_global_stack_traces_enabled,
    start_profiling, stop_profiling, ExportFormat, MemoryProfiler, MetricsScope,
};

/// Benchmark: Baseline computation without any profiling
fn bench_baseline_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline");

    group.bench_function("no_profiling", |b| {
        b.iter(|| {
            // Pure computation without profiling
            let mut sum = 0u64;
            for i in 0..1000 {
                sum = sum.wrapping_add(black_box(i * i));
            }
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark: Same computation with minimal profiling
fn bench_minimal_profiling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("minimal_profiling");

    group.bench_function("with_profiling", |b| {
        b.iter(|| {
            start_profiling();
            {
                profile_scope!("computation");
                let mut sum = 0u64;
                for i in 0..1000 {
                    sum = sum.wrapping_add(black_box(i * i));
                }
                black_box(sum);
            }
            stop_profiling();
            clear_global_events();
        });
    });

    group.finish();
}

/// Benchmark: Computation with detailed metrics collection
fn bench_detailed_profiling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("detailed_profiling");

    group.bench_function("with_metrics", |b| {
        b.iter(|| {
            start_profiling();
            {
                let mut scope = MetricsScope::new("computation_with_metrics");
                scope.set_operation_count(1000);
                scope.set_flops(5000);
                scope.set_bytes_transferred(8000);

                let mut sum = 0u64;
                for i in 0..1000 {
                    sum = sum.wrapping_add(black_box(i * i));
                }
                black_box(sum);
            }
            stop_profiling();
            clear_global_events();
        });
    });

    group.finish();
}

/// Benchmark: Stack trace capture overhead
fn bench_stack_trace_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("stack_traces");

    group.bench_function("without_stack_traces", |b| {
        set_global_stack_traces_enabled(false);
        b.iter(|| {
            start_profiling();
            {
                profile_scope!("operation");
                black_box(42);
            }
            stop_profiling();
            clear_global_events();
        });
    });

    group.bench_function("with_stack_traces", |b| {
        set_global_stack_traces_enabled(true);
        b.iter(|| {
            start_profiling();
            {
                profile_scope!("operation");
                black_box(42);
            }
            stop_profiling();
            clear_global_events();
        });
        set_global_stack_traces_enabled(false);
    });

    group.finish();
}

/// Benchmark: Nested scope overhead at different depths
fn bench_nested_scope_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_scopes");

    for depth in [1, 5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*depth as u64));
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            b.iter(|| {
                start_profiling();
                nest_scopes(black_box(depth));
                stop_profiling();
                clear_global_events();
            });
        });
    }

    group.finish();
}

/// Helper function to create nested scopes recursively
fn nest_scopes(depth: usize) {
    if depth == 0 {
        black_box(42);
    } else {
        profile_scope!(format!("level_{}", depth));
        nest_scopes(depth - 1);
    }
}

/// Benchmark: Event volume impact
fn bench_event_volume_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_volume");

    for num_events in [10, 100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*num_events as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_events),
            num_events,
            |b, &num_events| {
                b.iter(|| {
                    start_profiling();
                    for i in 0..num_events {
                        profile_scope!(format!("event_{}", i));
                    }
                    stop_profiling();
                    clear_global_events();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory profiling overhead
fn bench_memory_profiling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_profiling");

    group.bench_function("without_memory_tracking", |b| {
        b.iter(|| {
            let mut data = Vec::new();
            for i in 0..100 {
                data.push(black_box(vec![i; 1000]));
            }
            black_box(data);
        });
    });

    group.bench_function("with_memory_tracking", |b| {
        b.iter(|| {
            let mut profiler = MemoryProfiler::new();
            profiler.enable();

            let mut data = Vec::new();
            for i in 0..100 {
                data.push(black_box(vec![i; 1000]));
                // Simulate allocation tracking
                profiler.record_allocation(
                    i * 1000,
                    1000 * std::mem::size_of::<i32>(),
                    "allocation",
                );
            }

            profiler.disable();
            black_box(data);
        });
    });

    group.finish();
}

/// Benchmark: Concurrent profiling contention
fn bench_concurrent_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_contention");

    for num_threads in [1, 2, 4, 8, 16].iter() {
        group.throughput(Throughput::Elements(*num_threads as u64 * 100));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    start_profiling();
                    let mut handles = vec![];

                    for thread_id in 0..num_threads {
                        let handle = thread::spawn(move || {
                            for i in 0..100 {
                                profile_scope!(format!("thread_{}_op_{}", thread_id, i));
                                // Minimal work to focus on profiling overhead
                                black_box(i * i);
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    stop_profiling();
                    clear_global_events();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Export format performance comparison
fn bench_export_format_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("export_formats");

    // Setup: Create a consistent set of events for export
    let setup_events = || {
        start_profiling();
        for i in 0..500 {
            profile_scope!(format!("operation_{}", i));
        }
        stop_profiling();
    };

    // Benchmark JSON export
    setup_events();
    group.bench_function("json", |b| {
        b.iter(|| {
            let _ = export_global_events(
                black_box(ExportFormat::Json),
                black_box("/tmp/bench_json.json"),
            );
        });
    });
    std::fs::remove_file("/tmp/bench_json.json").ok();
    clear_global_events();

    // Benchmark CSV export
    setup_events();
    group.bench_function("csv", |b| {
        b.iter(|| {
            let _ = export_global_events(
                black_box(ExportFormat::Csv),
                black_box("/tmp/bench_csv.csv"),
            );
        });
    });
    std::fs::remove_file("/tmp/bench_csv.csv").ok();
    clear_global_events();

    // Benchmark Chrome Trace export
    setup_events();
    group.bench_function("chrome_trace", |b| {
        b.iter(|| {
            let _ = export_global_events(
                black_box(ExportFormat::ChromeTrace),
                black_box("/tmp/bench_chrome.json"),
            );
        });
    });
    std::fs::remove_file("/tmp/bench_chrome.json").ok();
    clear_global_events();

    group.finish();
}

/// Benchmark: Profiling overhead percentage calculation
fn bench_overhead_percentage(c: &mut Criterion) {
    let mut group = c.benchmark_group("overhead_percentage");

    // Simulate different workload durations
    for workload_ms in [1, 10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("baseline", workload_ms),
            workload_ms,
            |b, &workload_ms| {
                b.iter(|| {
                    thread::sleep(Duration::from_millis(black_box(workload_ms)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("with_profiling", workload_ms),
            workload_ms,
            |b, &workload_ms| {
                b.iter(|| {
                    start_profiling();
                    {
                        profile_scope!("workload");
                        thread::sleep(Duration::from_millis(black_box(workload_ms)));
                    }
                    stop_profiling();
                    clear_global_events();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_baseline_computation,
    bench_minimal_profiling_overhead,
    bench_detailed_profiling_overhead,
    bench_stack_trace_overhead,
    bench_nested_scope_overhead,
    bench_event_volume_overhead,
    bench_memory_profiling_overhead,
    bench_concurrent_contention,
    bench_export_format_comparison,
    bench_overhead_percentage,
);

criterion_main!(benches);
