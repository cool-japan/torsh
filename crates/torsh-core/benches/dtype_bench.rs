use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use torsh_core::dtype::DType;

fn dtype_creation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_creation");

    // Benchmark dtype creation for different types
    group.bench_function("create_f32", |b| b.iter(|| DType::F32));
    group.bench_function("create_f64", |b| b.iter(|| DType::F64));
    group.bench_function("create_i32", |b| b.iter(|| DType::I32));
    group.bench_function("create_i64", |b| b.iter(|| DType::I64));
    group.bench_function("create_bool", |b| b.iter(|| DType::Bool));
    group.bench_function("create_u8", |b| b.iter(|| DType::U8));
    group.bench_function("create_i8", |b| b.iter(|| DType::I8));
    group.bench_function("create_i16", |b| b.iter(|| DType::I16));
    group.bench_function("create_qint8", |b| b.iter(|| DType::QInt8));
    group.bench_function("create_quint8", |b| b.iter(|| DType::QUInt8));

    #[cfg(feature = "half")]
    {
        group.bench_function("create_f16", |b| b.iter(|| DType::F16));
        group.bench_function("create_bf16", |b| b.iter(|| DType::BF16));
    }

    group.bench_function("create_c64", |b| b.iter(|| DType::C64));
    group.bench_function("create_c128", |b| b.iter(|| DType::C128));

    group.finish();
}

fn dtype_properties_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_properties");

    let dtypes = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::U8,
        DType::I8,
        DType::I16,
        DType::BF16,
        DType::F16,
        DType::C64,
        DType::C128,
        DType::QInt8,
        DType::QUInt8,
    ];

    // Benchmark size_of operations
    for (i, dtype) in dtypes.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("size_of", i), dtype, |b, dtype| {
            b.iter(|| dtype.size())
        });

        group.bench_with_input(
            BenchmarkId::new("is_floating_point", i),
            dtype,
            |b, dtype| b.iter(|| dtype.is_float()),
        );

        group.bench_with_input(BenchmarkId::new("is_integer", i), dtype, |b, dtype| {
            b.iter(|| dtype.is_int())
        });

        group.bench_with_input(BenchmarkId::new("is_signed", i), dtype, |b, dtype| {
            b.iter(|| {
                dtype.is_float()
                    || matches!(
                        dtype,
                        DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::QInt8
                    )
            })
        });

        group.bench_with_input(BenchmarkId::new("is_complex", i), dtype, |b, dtype| {
            b.iter(|| dtype.is_complex())
        });
    }

    group.finish();
}

fn dtype_comparison_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_comparison");

    let f32_type = DType::F32;
    let f64_type = DType::F64;
    let i32_type = DType::I32;
    let _bool_type = DType::Bool;

    // Benchmark equality comparisons
    group.bench_function("equal_f32", |b| {
        b.iter(|| std::hint::black_box(f32_type) == std::hint::black_box(f32_type))
    });

    group.bench_function("different_types", |b| b.iter(|| f32_type == i32_type));

    group.bench_function("floating_vs_integer", |b| b.iter(|| f64_type == i32_type));

    // Benchmark hashing
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    group.bench_function("hash_f32", |b| {
        b.iter(|| {
            let mut hasher = DefaultHasher::new();
            f32_type.hash(&mut hasher);
            hasher.finish()
        })
    });

    group.bench_function("hash_complex", |b| {
        b.iter(|| {
            let mut hasher = DefaultHasher::new();
            DType::C64.hash(&mut hasher);
            hasher.finish()
        })
    });

    group.finish();
}

fn dtype_size_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_sizes");

    // Test dtype size calculations
    let dtypes = [
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::BF16,
        DType::C64,
        DType::C128,
        DType::Bool,
        DType::QInt8,
        DType::QUInt8,
    ];

    for (i, dtype) in dtypes.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("dtype_size", i), dtype, |b, dtype| {
            b.iter(|| dtype.size())
        });
    }

    // Benchmark size comparisons
    group.bench_function("compare_sizes", |b| {
        b.iter(|| dtypes.iter().map(|dt| dt.size()).max())
    });

    group.finish();
}

fn dtype_conversion_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_conversion");
    group.throughput(Throughput::Elements(1000));

    // Benchmark string conversions
    let dtypes = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C64,
    ];

    for (i, dtype) in dtypes.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("to_string", i), dtype, |b, dtype| {
            b.iter(|| dtype.to_string())
        });

        group.bench_with_input(BenchmarkId::new("debug_format", i), dtype, |b, dtype| {
            b.iter(|| format!("{:?}", dtype))
        });

        group.bench_with_input(BenchmarkId::new("display_format", i), dtype, |b, dtype| {
            b.iter(|| format!("{}", dtype))
        });
    }

    // Benchmark dtype names
    group.bench_function("dtype_name_f32", |b| b.iter(|| format!("{:?}", DType::F32)));

    group.bench_function("dtype_name_f64", |b| b.iter(|| format!("{:?}", DType::F64)));

    group.bench_function("dtype_name_i32", |b| b.iter(|| format!("{:?}", DType::I32)));

    group.bench_function("dtype_name_bool", |b| {
        b.iter(|| format!("{:?}", DType::Bool))
    });

    group.finish();
}

fn dtype_arrays_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_arrays");
    group.throughput(Throughput::Elements(100));

    // Benchmark operations on arrays of dtypes
    let dtypes = [DType::F32; 100];
    let mixed_dtypes = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::U8,
        DType::I8,
        DType::C64,
        DType::C128,
    ];

    group.bench_function("check_all_same", |b| {
        b.iter(|| dtypes.iter().all(|&dt| dt == DType::F32))
    });

    group.bench_function("check_all_floating", |b| {
        b.iter(|| dtypes.iter().all(|dt| dt.is_float()))
    });

    group.bench_function("find_largest_dtype", |b| {
        b.iter(|| mixed_dtypes.iter().max_by_key(|dt| dt.size()).unwrap())
    });

    group.bench_function("count_floating_types", |b| {
        b.iter(|| mixed_dtypes.iter().filter(|dt| dt.is_float()).count())
    });

    group.finish();
}

fn dtype_memory_layout_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_memory_layout");

    // Benchmark size-related operations
    let dtypes = [
        DType::Bool,
        DType::U8,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::F32,
        DType::I64,
        DType::F64,
        DType::C64,
        DType::C128,
    ];

    for (i, dtype) in dtypes.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("size_bytes", i), dtype, |b, dtype| {
            b.iter(|| dtype.size_bytes())
        });

        group.bench_with_input(BenchmarkId::new("name", i), dtype, |b, dtype| {
            b.iter(|| format!("{:?}", dtype))
        });
    }

    // Benchmark size calculations for arrays
    group.bench_function("calculate_array_size_1000", |b| {
        b.iter(|| {
            let dtype = DType::F32;
            dtype.size() * 1000
        })
    });

    group.bench_function("calculate_matrix_size_32x32", |b| {
        b.iter(|| {
            let dtype = DType::F64;
            dtype.size() * 32 * 32
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    dtype_creation_benchmarks,
    dtype_properties_benchmarks,
    dtype_comparison_benchmarks,
    dtype_size_benchmarks,
    dtype_conversion_benchmarks,
    dtype_arrays_benchmarks,
    dtype_memory_layout_benchmarks
);
criterion_main!(benches);
