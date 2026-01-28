//! Comprehensive benchmarks for torsh-profiler
//!
//! This benchmark suite measures the performance of various profiling operations
//! to ensure minimal overhead in production environments.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::thread;
use torsh_profiler::{
    add_global_event, clear_global_events, export_global_events, profile_scope, start_profiling,
    stop_profiling, ExportFormat, MetricsScope, ScopeGuard,
};

/// Benchmark basic event recording
fn bench_add_event(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_event");

    for size in [1, 10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                start_profiling();
                for i in 0..size {
                    add_global_event(
                        &format!("operation_{}", i),
                        "benchmark",
                        black_box(100),
                        black_box(1),
                    );
                }
                stop_profiling();
                clear_global_events();
            });
        });
    }

    group.finish();
}

/// Benchmark scope guard creation and destruction
fn bench_scope_guard(c: &mut Criterion) {
    let mut group = c.benchmark_group("scope_guard");

    group.bench_function("basic_scope", |b| {
        b.iter(|| {
            start_profiling();
            {
                let _guard = ScopeGuard::new(black_box("benchmark_scope"));
                // Simulate some work
                black_box(42);
            }
            stop_profiling();
            clear_global_events();
        });
    });

    group.bench_function("nested_scopes", |b| {
        b.iter(|| {
            start_profiling();
            {
                let _guard1 = ScopeGuard::new(black_box("outer"));
                {
                    let _guard2 = ScopeGuard::new(black_box("middle"));
                    {
                        let _guard3 = ScopeGuard::new(black_box("inner"));
                        black_box(42);
                    }
                }
            }
            stop_profiling();
            clear_global_events();
        });
    });

    group.finish();
}

/// Benchmark metrics scope with various metrics
fn bench_metrics_scope(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_scope");

    group.bench_function("full_metrics", |b| {
        b.iter(|| {
            start_profiling();
            {
                let mut scope = MetricsScope::new(black_box("benchmark"));
                scope.set_operation_count(black_box(1000));
                scope.set_flops(black_box(50000));
                scope.set_bytes_transferred(black_box(4096));
                black_box(&scope);
            }
            stop_profiling();
            clear_global_events();
        });
    });

    group.bench_function("minimal_metrics", |b| {
        b.iter(|| {
            start_profiling();
            {
                let scope = MetricsScope::new(black_box("benchmark"));
                black_box(&scope);
            }
            stop_profiling();
            clear_global_events();
        });
    });

    group.finish();
}

/// Benchmark export operations
fn bench_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("export");

    // Setup: Create events
    start_profiling();
    for i in 0..100 {
        add_global_event(&format!("operation_{}", i), "benchmark", 100, 1);
    }
    stop_profiling();

    group.bench_function("json_export", |b| {
        b.iter(|| {
            let _ = export_global_events(
                black_box(ExportFormat::Json),
                black_box("/tmp/bench_export.json"),
            );
        });
    });

    group.bench_function("csv_export", |b| {
        b.iter(|| {
            let _ = export_global_events(
                black_box(ExportFormat::Csv),
                black_box("/tmp/bench_export.csv"),
            );
        });
    });

    group.bench_function("chrome_trace_export", |b| {
        b.iter(|| {
            let _ = export_global_events(
                black_box(ExportFormat::ChromeTrace),
                black_box("/tmp/bench_export_trace.json"),
            );
        });
    });

    group.finish();

    // Cleanup
    let _ = std::fs::remove_file("/tmp/bench_export.json");
    let _ = std::fs::remove_file("/tmp/bench_export.csv");
    let _ = std::fs::remove_file("/tmp/bench_export_trace.json");
    clear_global_events();
}

/// Benchmark profiler overhead with concurrent operations
fn bench_concurrent_profiling(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent");

    for num_threads in [1, 2, 4, 8].iter() {
        group.throughput(Throughput::Elements(*num_threads as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    start_profiling();
                    let mut handles = vec![];

                    for thread_id in 0..num_threads {
                        let handle = thread::spawn(move || {
                            for i in 0..10 {
                                add_global_event(
                                    &format!("thread_{}_op_{}", thread_id, i),
                                    "concurrent",
                                    black_box(100),
                                    black_box(thread_id),
                                );
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

/// Benchmark macro overhead
fn bench_macro_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("macros");

    group.bench_function("profile_scope_macro", |b| {
        b.iter(|| {
            start_profiling();
            profile_scope!("benchmark_macro");
            black_box(42);
            stop_profiling();
            clear_global_events();
        });
    });

    group.finish();
}

/// Benchmark profiler statistics collection
fn bench_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics");

    // Setup
    start_profiling();
    for i in 0..1000 {
        add_global_event(
            &format!("operation_{}", i),
            "stats",
            100 + (i % 50),
            (i % 4) as usize,
        );
    }
    stop_profiling();

    group.bench_function("get_stats", |b| {
        b.iter(|| {
            use torsh_profiler::get_global_stats;
            let stats = get_global_stats();
            let _ = black_box(stats);
        });
    });

    group.finish();
    clear_global_events();
}

/// Benchmark stack trace capture (expensive operation)
fn bench_stack_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("stack_trace");

    group.bench_function("with_stack_trace", |b| {
        use torsh_profiler::set_global_stack_traces_enabled;

        b.iter(|| {
            set_global_stack_traces_enabled(true);
            start_profiling();
            {
                let _guard = ScopeGuard::new(black_box("with_trace"));
                black_box(42);
            }
            stop_profiling();
            set_global_stack_traces_enabled(false);
            clear_global_events();
        });
    });

    group.bench_function("without_stack_trace", |b| {
        use torsh_profiler::set_global_stack_traces_enabled;

        b.iter(|| {
            set_global_stack_traces_enabled(false);
            start_profiling();
            {
                let _guard = ScopeGuard::new(black_box("without_trace"));
                black_box(42);
            }
            stop_profiling();
            clear_global_events();
        });
    });

    group.finish();
}

/// Benchmark overhead tracking
fn bench_overhead_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("overhead_tracking");

    group.bench_function("with_overhead_tracking", |b| {
        use torsh_profiler::{reset_global_overhead_stats, set_global_overhead_tracking_enabled};

        b.iter(|| {
            set_global_overhead_tracking_enabled(true);
            start_profiling();
            for i in 0..10 {
                add_global_event(
                    &format!("op_{}", i),
                    "overhead",
                    black_box(100),
                    black_box(1),
                );
            }
            stop_profiling();
            set_global_overhead_tracking_enabled(false);
            reset_global_overhead_stats();
            clear_global_events();
        });
    });

    group.bench_function("without_overhead_tracking", |b| {
        use torsh_profiler::set_global_overhead_tracking_enabled;

        b.iter(|| {
            set_global_overhead_tracking_enabled(false);
            start_profiling();
            for i in 0..10 {
                add_global_event(
                    &format!("op_{}", i),
                    "overhead",
                    black_box(100),
                    black_box(1),
                );
            }
            stop_profiling();
            clear_global_events();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_add_event,
    bench_scope_guard,
    bench_metrics_scope,
    bench_export,
    bench_concurrent_profiling,
    bench_macro_overhead,
    bench_statistics,
    bench_stack_trace,
    bench_overhead_tracking,
);

criterion_main!(benches);
