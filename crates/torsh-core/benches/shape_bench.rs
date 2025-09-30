use criterion::{criterion_group, criterion_main, Criterion};
use torsh_core::shape::Shape;

fn shape_creation_benchmarks(c: &mut Criterion) {
    c.bench_function("create_small_shape", |b| {
        b.iter(|| Shape::new(vec![32, 32]))
    });

    c.bench_function("create_medium_shape", |b| {
        b.iter(|| Shape::new(vec![128, 128, 64]))
    });

    c.bench_function("create_large_shape", |b| {
        b.iter(|| Shape::new(vec![256, 256, 64, 32]))
    });
}

fn shape_operations_benchmarks(c: &mut Criterion) {
    let shape = Shape::new(vec![32, 32]);

    c.bench_function("numel", |b| b.iter(|| shape.numel()));

    c.bench_function("ndim", |b| b.iter(|| shape.ndim()));

    c.bench_function("default_strides", |b| b.iter(|| shape.default_strides()));
}

criterion_group!(
    benches,
    shape_creation_benchmarks,
    shape_operations_benchmarks
);
criterion_main!(benches);
