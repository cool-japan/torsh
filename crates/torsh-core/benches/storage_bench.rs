//! Storage Performance Benchmarks
//!
//! Comprehensive benchmarks for storage operations including:
//! - Memory allocation and deallocation
//! - Data copying patterns
//! - Sequential vs random access patterns
//! - Cache effects
//!
//! # Running Benchmarks
//! ```bash
//! cargo bench --bench storage_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Benchmark memory allocation patterns
fn bench_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    for size in [1024, 4096, 16384, 65536, 262144, 1048576].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("vec_f32", size), size, |b, &size| {
            let elements = size / std::mem::size_of::<f32>();
            b.iter(|| {
                let data: Vec<f32> = Vec::with_capacity(elements);
                black_box(data);
            });
        });

        group.bench_with_input(BenchmarkId::new("vec_f64", size), size, |b, &size| {
            let elements = size / std::mem::size_of::<f64>();
            b.iter(|| {
                let data: Vec<f64> = Vec::with_capacity(elements);
                black_box(data);
            });
        });

        group.bench_with_input(BenchmarkId::new("vec_with_init", size), size, |b, &size| {
            let elements = size / std::mem::size_of::<f32>();
            b.iter(|| {
                let data: Vec<f32> = vec![0.0; elements];
                black_box(data);
            });
        });
    }

    group.finish();
}

/// Benchmark memory copying operations
fn bench_memory_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_copy");

    for size in [1024, 16384, 262144, 1048576].iter() {
        let elements = size / std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(*size as u64));

        let source: Vec<f32> = (0..elements).map(|i| i as f32).collect();

        group.bench_with_input(BenchmarkId::new("copy_f32", size), size, |b, _| {
            let mut dest: Vec<f32> = Vec::with_capacity(elements);
            b.iter(|| {
                dest.clear();
                dest.extend_from_slice(&source);
                black_box(&dest);
            });
        });

        group.bench_with_input(BenchmarkId::new("clone_vec", size), size, |b, _| {
            b.iter(|| {
                let cloned = source.clone();
                black_box(cloned);
            });
        });
    }

    group.finish();
}

/// Benchmark sequential vs random access
fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_patterns");

    let size = 65536;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    group.bench_function("sequential_read", |b| {
        b.iter(|| {
            let sum: f32 = data.iter().sum();
            black_box(sum);
        });
    });

    // Strided access (every 128th element)
    let indices: Vec<usize> = (0..size).step_by(128).collect();
    group.bench_function("strided_read_128", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &idx in &indices {
                sum += data[idx];
            }
            black_box(sum);
        });
    });

    // Random-like access pattern
    let random_indices: Vec<usize> = (0..512).map(|i| (i * 127) % size).collect();
    group.bench_function("pseudo_random_read", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &idx in &random_indices {
                sum += data[idx];
            }
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark memory fill operations
fn bench_memory_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_fill");

    for size in [1024, 16384, 262144, 1048576].iter() {
        let elements = size / std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("fill_zeros", size), size, |b, _| {
            let mut data: Vec<f32> = Vec::with_capacity(elements);
            data.resize(elements, 0.0);
            b.iter(|| {
                data.fill(0.0);
                black_box(&data);
            });
        });

        group.bench_with_input(BenchmarkId::new("fill_value", size), size, |b, _| {
            let mut data: Vec<f32> = Vec::with_capacity(elements);
            data.resize(elements, 0.0);
            b.iter(|| {
                data.fill(42.0);
                black_box(&data);
            });
        });
    }

    group.finish();
}

/// Benchmark storage creation patterns
fn bench_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_creation");

    let size = 16384;

    group.bench_function("vec_new", |b| {
        b.iter(|| {
            let data: Vec<f32> = Vec::new();
            black_box(data);
        });
    });

    group.bench_function("vec_with_capacity", |b| {
        b.iter(|| {
            let data: Vec<f32> = Vec::with_capacity(size);
            black_box(data);
        });
    });

    group.bench_function("vec_macro", |b| {
        b.iter(|| {
            let data: Vec<f32> = vec![1.0; size];
            black_box(data);
        });
    });

    group.bench_function("vec_from_iterator", |b| {
        b.iter(|| {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            black_box(data);
        });
    });

    group.finish();
}

/// Benchmark data type conversions
fn bench_type_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_conversions");

    let size = 16384;
    let f32_data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    group.bench_function("f32_to_f64", |b| {
        b.iter(|| {
            let f64_data: Vec<f64> = f32_data.iter().map(|&x| x as f64).collect();
            black_box(f64_data);
        });
    });

    let f64_data: Vec<f64> = (0..size).map(|i| i as f64).collect();

    group.bench_function("f64_to_f32", |b| {
        b.iter(|| {
            let f32_data: Vec<f32> = f64_data.iter().map(|&x| x as f32).collect();
            black_box(f32_data);
        });
    });

    group.finish();
}

/// Benchmark cache effects
fn bench_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effects");

    // L1 cache friendly (32KB typical)
    let small_size = 8192 / std::mem::size_of::<f32>();
    let small_data: Vec<f32> = (0..small_size).map(|i| i as f32).collect();

    // L3 cache unfriendly (4MB)
    let large_size = 4_194_304 / std::mem::size_of::<f32>();
    let large_data: Vec<f32> = (0..large_size).map(|i| i as f32).collect();

    group.bench_function("cache_friendly_small", |b| {
        b.iter(|| {
            let sum: f32 = small_data.iter().sum();
            black_box(sum);
        });
    });

    group.bench_function("cache_unfriendly_large", |b| {
        b.iter(|| {
            let sum: f32 = large_data.iter().sum();
            black_box(sum);
        });
    });

    // Cache line effects
    let cache_line_data: Vec<f32> = vec![1.0; 256]; // 1KB, ~16 cache lines

    group.bench_function("single_cache_line", |b| {
        b.iter(|| {
            // Access first 16 elements (one cache line)
            let sum: f32 = cache_line_data[0..16].iter().sum();
            black_box(sum);
        });
    });

    group.bench_function("multiple_cache_lines", |b| {
        b.iter(|| {
            // Access all elements (multiple cache lines)
            let sum: f32 = cache_line_data.iter().sum();
            black_box(sum);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_allocation,
    bench_memory_copy,
    bench_access_patterns,
    bench_memory_fill,
    bench_creation,
    bench_type_conversions,
    bench_cache_effects
);
criterion_main!(benches);
