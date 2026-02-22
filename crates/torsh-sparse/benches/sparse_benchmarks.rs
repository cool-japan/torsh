//! Performance benchmarks for sparse tensor operations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use torsh_core::Shape;
use torsh_sparse::ops::{norm, scale, spadd, sphadamard, spmm, sum, sum_axis, transpose};
use torsh_sparse::{CooTensor, CscTensor, CsrTensor, SparseTensor};
use torsh_tensor::creation::ones;

fn create_sparse_matrix(rows: usize, cols: usize, density: f32) -> CooTensor {
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    let nnz = ((rows * cols) as f32 * density) as usize;

    for _ in 0..nnz {
        let row = fastrand::usize(0..rows);
        let col = fastrand::usize(0..cols);
        let val = fastrand::f32() * 2.0 - 1.0; // Random value between -1 and 1

        row_indices.push(row);
        col_indices.push(col);
        values.push(val);
    }

    CooTensor::new(
        row_indices,
        col_indices,
        values,
        Shape::new(vec![rows, cols]),
    )
    .unwrap()
}

fn bench_sparse_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_creation");

    for size in [100, 500, 1000, 2000].iter() {
        let density = 0.1;
        let nnz = ((*size * *size) as f32 * density) as usize;

        // Generate random data
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for _ in 0..nnz {
            row_indices.push(fastrand::usize(0..*size));
            col_indices.push(fastrand::usize(0..*size));
            values.push(fastrand::f32());
        }

        let shape = Shape::new(vec![*size, *size]);

        group.bench_with_input(BenchmarkId::new("COO", size), size, |b, _| {
            b.iter(|| {
                CooTensor::new(
                    black_box(row_indices.clone()),
                    black_box(col_indices.clone()),
                    black_box(values.clone()),
                    black_box(shape.clone()),
                )
            })
        });

        // Benchmark CSR creation from COO
        let coo = CooTensor::new(
            row_indices.clone(),
            col_indices.clone(),
            values.clone(),
            shape.clone(),
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::new("CSR_from_COO", size), size, |b, _| {
            b.iter(|| CsrTensor::from_coo(black_box(&coo)))
        });

        // Benchmark CSC creation from COO
        group.bench_with_input(BenchmarkId::new("CSC_from_COO", size), size, |b, _| {
            b.iter(|| CscTensor::from_coo(black_box(&coo)))
        });
    }

    group.finish();
}

fn bench_sparse_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_conversion");

    for size in [100, 500, 1000].iter() {
        let density = 0.1;
        let sparse_coo = create_sparse_matrix(*size, *size, density);
        let sparse_csr = CsrTensor::from_coo(&sparse_coo).unwrap();
        let sparse_csc = CscTensor::from_coo(&sparse_coo).unwrap();

        group.bench_with_input(BenchmarkId::new("COO_to_CSR", size), size, |b, _| {
            b.iter(|| sparse_coo.to_csr())
        });

        group.bench_with_input(BenchmarkId::new("COO_to_CSC", size), size, |b, _| {
            b.iter(|| sparse_coo.to_csc())
        });

        group.bench_with_input(BenchmarkId::new("CSR_to_COO", size), size, |b, _| {
            b.iter(|| sparse_csr.to_coo())
        });

        group.bench_with_input(BenchmarkId::new("CSC_to_COO", size), size, |b, _| {
            b.iter(|| sparse_csc.to_coo())
        });

        group.bench_with_input(BenchmarkId::new("COO_to_dense", size), size, |b, _| {
            b.iter(|| sparse_coo.to_dense())
        });
    }

    group.finish();
}

fn bench_sparse_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_operations");

    for size in [100, 500, 1000].iter() {
        let density = 0.1;
        let sparse_a = create_sparse_matrix(*size, *size, density);
        let sparse_b = create_sparse_matrix(*size, *size, density);
        let dense_b = ones::<f32>(&[*size, *size]).unwrap();

        group.bench_with_input(BenchmarkId::new("spmm", size), size, |b, _| {
            b.iter(|| {
                spmm(
                    black_box(&sparse_a as &dyn SparseTensor),
                    black_box(&dense_b),
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("spadd", size), size, |b, _| {
            b.iter(|| {
                spadd(
                    black_box(&sparse_a as &dyn SparseTensor),
                    black_box(&sparse_b as &dyn SparseTensor),
                    black_box(1.0),
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("sphadamard", size), size, |b, _| {
            b.iter(|| {
                sphadamard(
                    black_box(&sparse_a as &dyn SparseTensor),
                    black_box(&sparse_b as &dyn SparseTensor),
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("transpose", size), size, |b, _| {
            b.iter(|| transpose(black_box(&sparse_a as &dyn SparseTensor)))
        });

        group.bench_with_input(BenchmarkId::new("sum", size), size, |b, _| {
            b.iter(|| sum(black_box(&sparse_a as &dyn SparseTensor)))
        });

        group.bench_with_input(BenchmarkId::new("sum_axis_0", size), size, |b, _| {
            b.iter(|| sum_axis(black_box(&sparse_a as &dyn SparseTensor), black_box(0)))
        });

        group.bench_with_input(BenchmarkId::new("norm", size), size, |b, _| {
            b.iter(|| norm(black_box(&sparse_a as &dyn SparseTensor)))
        });

        group.bench_with_input(BenchmarkId::new("scale", size), size, |b, _| {
            b.iter(|| scale(black_box(&sparse_a as &dyn SparseTensor), black_box(2.0)))
        });
    }

    group.finish();
}

fn bench_sparse_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matvec");

    for size in [100, 500, 1000, 2000].iter() {
        let density = 0.1;
        let sparse_coo = create_sparse_matrix(*size, *size, density);
        let sparse_csr = CsrTensor::from_coo(&sparse_coo).unwrap();
        let vector = ones::<f32>(&[*size]).unwrap();

        group.bench_with_input(BenchmarkId::new("CSR_matvec", size), size, |b, _| {
            b.iter(|| sparse_csr.matvec(black_box(&vector)))
        });
    }

    group.finish();
}

fn bench_format_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_comparison");

    let size = 1000;
    let density = 0.1;
    let sparse_coo = create_sparse_matrix(size, size, density);
    let sparse_csr = CsrTensor::from_coo(&sparse_coo).unwrap();
    let sparse_csc = CscTensor::from_coo(&sparse_coo).unwrap();
    let dense_matrix = ones::<f32>(&[size, size]).unwrap();
    let vector = ones::<f32>(&[size]).unwrap();

    // Compare sparse-dense multiplication performance
    group.bench_function("COO_spmm", |b| {
        b.iter(|| {
            spmm(
                black_box(&sparse_coo as &dyn SparseTensor),
                black_box(&dense_matrix),
            )
        })
    });

    group.bench_function("CSR_spmm", |b| {
        b.iter(|| {
            spmm(
                black_box(&sparse_csr as &dyn SparseTensor),
                black_box(&dense_matrix),
            )
        })
    });

    group.bench_function("CSC_spmm", |b| {
        b.iter(|| {
            spmm(
                black_box(&sparse_csc as &dyn SparseTensor),
                black_box(&dense_matrix),
            )
        })
    });

    // Compare matrix-vector multiplication
    group.bench_function("CSR_matvec", |b| {
        b.iter(|| sparse_csr.matvec(black_box(&vector)))
    });

    group.finish();
}

fn bench_sparsity_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsity_effects");

    let size = 500;
    for density in [0.01, 0.05, 0.1, 0.2, 0.5].iter() {
        let sparse = create_sparse_matrix(size, size, *density);
        let dense_matrix = ones::<f32>(&[size, size]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("spmm", (density * 100.0) as usize),
            density,
            |b, _| {
                b.iter(|| {
                    spmm(
                        black_box(&sparse as &dyn SparseTensor),
                        black_box(&dense_matrix),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sum", (density * 100.0) as usize),
            density,
            |b, _| b.iter(|| sum(black_box(&sparse as &dyn SparseTensor))),
        );

        group.bench_with_input(
            BenchmarkId::new("norm", (density * 100.0) as usize),
            density,
            |b, _| b.iter(|| norm(black_box(&sparse as &dyn SparseTensor))),
        );
    }

    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Benchmark memory efficiency of different formats
    for size in [100, 500, 1000].iter() {
        let density = 0.1;
        let sparse_coo = create_sparse_matrix(*size, *size, density);

        group.bench_with_input(BenchmarkId::new("COO_creation", size), size, |b, _| {
            b.iter(|| {
                let triplets = sparse_coo.triplets();
                let (rows, cols, vals): (Vec<_>, Vec<_>, Vec<_>) = triplets.into_iter().fold(
                    (Vec::new(), Vec::new(), Vec::new()),
                    |(mut rows, mut cols, mut vals), (r, c, v)| {
                        rows.push(r);
                        cols.push(c);
                        vals.push(v);
                        (rows, cols, vals)
                    },
                );
                CooTensor::new(rows, cols, vals, sparse_coo.shape().clone())
            })
        });

        group.bench_with_input(BenchmarkId::new("CSR_creation", size), size, |b, _| {
            b.iter(|| CsrTensor::from_coo(black_box(&sparse_coo)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sparse_creation,
    bench_sparse_conversion,
    bench_sparse_operations,
    bench_sparse_matvec,
    bench_format_comparison,
    bench_sparsity_effects,
    bench_memory_usage
);
criterion_main!(benches);
