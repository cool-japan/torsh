use criterion::{black_box, criterion_group, criterion_main, Criterion};
use torsh_core::DeviceType;
use torsh_linalg::*;
use torsh_tensor::Tensor;

fn create_test_matrix(size: usize) -> Tensor {
    let mut data = vec![0.0f32; size * size];
    for i in 0..size {
        for j in 0..size {
            data[i * size + j] = ((i + j) as f32) / size as f32 + 1.0;
        }
    }
    Tensor::from_data(data, vec![size, size], DeviceType::Cpu).unwrap()
}

fn create_spd_matrix(size: usize) -> Tensor {
    let a = create_test_matrix(size);
    let at = a.transpose_view(0, 1).unwrap();
    matmul(&a, &at).unwrap()
}

fn bench_matrix_multiplication(c: &mut Criterion) {
    let sizes = vec![16, 32, 64, 128];

    for size in sizes {
        let a = create_test_matrix(size);
        let b = create_test_matrix(size);

        c.bench_function(&format!("matmul_{size}x{size}"), |bench| {
            bench.iter(|| {
                let _ = matmul(black_box(&a), black_box(&b));
            })
        });
    }
}

fn bench_decompositions(c: &mut Criterion) {
    let sizes = vec![16, 32, 64];

    for size in sizes {
        let matrix = create_test_matrix(size);
        let spd_matrix = create_spd_matrix(size);

        c.bench_function(&format!("lu_decomp_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = lu(black_box(&matrix));
            })
        });

        c.bench_function(&format!("qr_decomp_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = qr(black_box(&matrix));
            })
        });

        c.bench_function(&format!("svd_decomp_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = svd(black_box(&matrix), false);
            })
        });

        c.bench_function(&format!("cholesky_decomp_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = cholesky(black_box(&spd_matrix), false);
            })
        });
    }
}

fn bench_matrix_functions(c: &mut Criterion) {
    let sizes = vec![16, 32, 64];

    for size in sizes {
        let matrix = create_test_matrix(size);

        c.bench_function(&format!("matrix_norm_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = matrix_norm(black_box(&matrix), None);
            })
        });

        c.bench_function(&format!("matrix_exp_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = matrix_exp(black_box(&matrix));
            })
        });

        c.bench_function(&format!("matrix_inv_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = inv(black_box(&matrix));
            })
        });

        c.bench_function(&format!("determinant_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = det(black_box(&matrix));
            })
        });
    }
}

fn bench_solvers(c: &mut Criterion) {
    let sizes = vec![16, 32, 64];

    for size in sizes {
        let matrix = create_test_matrix(size);
        let vector = Tensor::from_data(
            (0..size).map(|i| i as f32 + 1.0).collect::<Vec<f32>>(),
            vec![size],
            DeviceType::Cpu,
        )
        .unwrap();

        c.bench_function(&format!("solve_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = solve(black_box(&matrix), black_box(&vector));
            })
        });

        c.bench_function(&format!("least_squares_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = lstsq(black_box(&matrix), black_box(&vector), None);
            })
        });
    }
}

fn bench_sparse_solvers(c: &mut Criterion) {
    let sizes = vec![16, 32, 64];

    for size in sizes {
        let spd_matrix = create_spd_matrix(size);
        let vector = Tensor::from_data(
            (0..size).map(|i| i as f32 + 1.0).collect::<Vec<f32>>(),
            vec![size],
            DeviceType::Cpu,
        )
        .unwrap();

        c.bench_function(&format!("cg_solver_{size}x{size}"), |b| {
            b.iter(|| {
                let _ = conjugate_gradient(
                    black_box(&spd_matrix),
                    black_box(&vector),
                    None,
                    Some(1e-6),
                    Some(1000),
                    None,
                );
            })
        });
    }
}

criterion_group!(
    benches,
    bench_matrix_multiplication,
    bench_decompositions,
    bench_matrix_functions,
    bench_solvers,
    bench_sparse_solvers
);
criterion_main!(benches);
