//! Comprehensive benchmarking suite for torsh-cluster
//!
//! This module provides Criterion benchmarks for all clustering algorithms,
//! comparing performance across different dataset sizes, dimensions, and configurations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use torsh_cluster::{
    algorithms::{
        dbscan::{HDBSCANConfig, DBSCAN, HDBSCAN},
        gaussian_mixture::{CovarianceType, GaussianMixture},
        hierarchical::{AgglomerativeClustering, Linkage},
        incremental::{IncrementalClustering, OnlineKMeans},
        kmeans::{KMeans, KMeansAlgorithm},
        optics::OPTICS,
        spectral::SpectralClustering,
    },
    traits::Fit,
};
use torsh_tensor::Tensor;

/// Generate synthetic clustered data for benchmarking
fn generate_clustered_data(n_samples: usize, n_features: usize, n_clusters: usize) -> Tensor {
    let mut data = Vec::with_capacity(n_samples * n_features);

    for cluster_id in 0..n_clusters {
        let samples_per_cluster = n_samples / n_clusters;
        for _ in 0..samples_per_cluster {
            for feature_id in 0..n_features {
                let base_value = (cluster_id * 10) as f32 + (feature_id as f32 * 0.1);
                let noise = ((cluster_id + feature_id) % 10) as f32 * 0.05;
                data.push(base_value + noise);
            }
        }
    }

    // Fill remaining samples if division isn't even
    let remaining = n_samples - (n_samples / n_clusters) * n_clusters;
    for _ in 0..remaining {
        for feature_id in 0..n_features {
            data.push(feature_id as f32 * 0.1);
        }
    }

    Tensor::from_vec(data, &[n_samples, n_features]).unwrap()
}

// ================================================================================================
// K-Means Benchmarks
// ================================================================================================

fn bench_kmeans_lloyd(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_lloyd");

    for size in [100, 500, 1000, 2000].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let kmeans = KMeans::new(5)
                .algorithm(KMeansAlgorithm::Lloyd)
                .max_iters(50)
                .random_state(42);

            b.iter(|| {
                kmeans.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_kmeans_elkan(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_elkan");

    for size in [100, 500, 1000, 2000].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let kmeans = KMeans::new(5)
                .algorithm(KMeansAlgorithm::Elkan)
                .max_iters(50)
                .random_state(42);

            b.iter(|| {
                kmeans.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_kmeans_minibatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_minibatch");

    for size in [100, 500, 1000, 2000].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let kmeans = KMeans::new(5)
                .algorithm(KMeansAlgorithm::MiniBatch)
                .max_iters(50)
                .random_state(42);

            b.iter(|| {
                kmeans.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_kmeans_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_algorithm_comparison");
    let data = generate_clustered_data(1000, 10, 5);

    for algorithm in [
        KMeansAlgorithm::Lloyd,
        KMeansAlgorithm::Elkan,
        KMeansAlgorithm::MiniBatch,
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", algorithm)),
            algorithm,
            |b, &algo| {
                let kmeans = KMeans::new(5)
                    .algorithm(algo)
                    .max_iters(50)
                    .random_state(42);

                b.iter(|| {
                    kmeans.fit(black_box(&data)).unwrap();
                });
            },
        );
    }

    group.finish();
}

// ================================================================================================
// DBSCAN Benchmarks
// ================================================================================================

fn bench_dbscan(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbscan");

    for size in [100, 200, 500].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let dbscan = DBSCAN::new(2.0, 3);

            b.iter(|| {
                dbscan.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

// ================================================================================================
// Hierarchical Clustering Benchmarks
// ================================================================================================

fn bench_hierarchical(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical");

    // Hierarchical is O(n^2) so use smaller sizes
    for size in [50, 100, 200].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let hierarchical = AgglomerativeClustering::new(5).linkage(Linkage::Average);

            b.iter(|| {
                hierarchical.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_hierarchical_linkages(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical_linkage_comparison");
    let data = generate_clustered_data(100, 10, 5);

    for linkage in [Linkage::Single, Linkage::Complete, Linkage::Average].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", linkage)),
            linkage,
            |b, &link| {
                let hierarchical = AgglomerativeClustering::new(5).linkage(link);

                b.iter(|| {
                    hierarchical.fit(black_box(&data)).unwrap();
                });
            },
        );
    }

    group.finish();
}

// ================================================================================================
// GMM Benchmarks
// ================================================================================================

fn bench_gmm_diagonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_diagonal");

    for size in [100, 500, 1000].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let gmm = GaussianMixture::new(5)
                .covariance_type(CovarianceType::Diag)
                .max_iters(50)
                .random_state(42);

            b.iter(|| {
                gmm.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_gmm_spherical(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_spherical");

    for size in [100, 500, 1000].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let gmm = GaussianMixture::new(5)
                .covariance_type(CovarianceType::Spherical)
                .max_iters(50)
                .random_state(42);

            b.iter(|| {
                gmm.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_gmm_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_full");

    // Full covariance is more expensive
    for size in [100, 500].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let gmm = GaussianMixture::new(5)
                .covariance_type(CovarianceType::Full)
                .max_iters(50)
                .random_state(42);

            b.iter(|| {
                gmm.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

// ================================================================================================
// Spectral Clustering Benchmarks
// ================================================================================================

fn bench_spectral(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral");

    // Spectral clustering is expensive due to eigendecomposition
    for size in [50, 100, 200].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let spectral = SpectralClustering::new(5);

            b.iter(|| {
                spectral.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

// ================================================================================================
// Scalability Benchmarks (varying dimensions)
// ================================================================================================

fn bench_kmeans_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_dimensions");

    for n_features in [5, 10, 20, 50].iter() {
        let data = generate_clustered_data(500, *n_features, 5);

        group.throughput(Throughput::Elements(500));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_features),
            n_features,
            |b, &_| {
                let kmeans = KMeans::new(5).max_iters(50).random_state(42);

                b.iter(|| {
                    kmeans.fit(black_box(&data)).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_gmm_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm_dimensions");

    for n_features in [5, 10, 20].iter() {
        let data = generate_clustered_data(200, *n_features, 5);

        group.throughput(Throughput::Elements(200));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_features),
            n_features,
            |b, &_| {
                let gmm = GaussianMixture::new(5)
                    .covariance_type(CovarianceType::Diag)
                    .max_iters(50)
                    .random_state(42);

                b.iter(|| {
                    gmm.fit(black_box(&data)).unwrap();
                });
            },
        );
    }

    group.finish();
}

// ================================================================================================
// Distance Computation Benchmarks (SIMD vs non-SIMD)
// ================================================================================================

fn bench_distance_computations(c: &mut Criterion) {
    use scirs2_core::ndarray::Array1;
    use torsh_cluster::utils::{
        batch_euclidean_distances_simd_f32, euclidean_distance_simd_f32,
        parallel_pairwise_distances_f32,
    };

    let mut group = c.benchmark_group("distance_computations");

    // Point-to-point distance
    let x = Array1::from_vec((0..100).map(|i| i as f32).collect());
    let y = Array1::from_vec((0..100).map(|i| (i + 1) as f32).collect());

    group.bench_function("euclidean_distance_simd", |b| {
        b.iter(|| {
            euclidean_distance_simd_f32(black_box(&x.view()), black_box(&y.view()));
        });
    });

    // Batch distance computation
    let points = (0..1000).map(|i| i as f32).collect::<Vec<_>>();
    let centroids = (0..50).map(|i| i as f32).collect::<Vec<_>>();

    group.bench_function("batch_distances_simd", |b| {
        b.iter(|| {
            batch_euclidean_distances_simd_f32(
                black_box(&points),
                black_box(&centroids),
                100,
                10,
                5,
            );
        });
    });

    // Parallel pairwise distances
    use scirs2_core::ndarray::Array2;
    let data = Array2::from_shape_vec((100, 10), (0..1000).map(|i| i as f32).collect()).unwrap();

    group.bench_function("parallel_pairwise_distances", |b| {
        b.iter(|| {
            parallel_pairwise_distances_f32(black_box(&data)).unwrap();
        });
    });

    group.finish();
}

// ================================================================================================
// Criterion configuration
// ================================================================================================

criterion_group!(
    kmeans_benches,
    bench_kmeans_lloyd,
    bench_kmeans_elkan,
    bench_kmeans_minibatch,
    bench_kmeans_comparison,
    bench_kmeans_dimensions,
);

criterion_group!(
    gmm_benches,
    bench_gmm_diagonal,
    bench_gmm_spherical,
    bench_gmm_full,
    bench_gmm_dimensions,
);

// ================================================================================================
// HDBSCAN Benchmarks
// ================================================================================================

fn bench_hdbscan(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdbscan");

    for size in [100, 200, 500].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let config = HDBSCANConfig {
                min_cluster_size: 5,
                min_samples: Some(3),
                ..Default::default()
            };
            let hdbscan = HDBSCAN::new(config);

            b.iter(|| {
                hdbscan.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

// ================================================================================================
// OPTICS Benchmarks
// ================================================================================================

fn bench_optics(c: &mut Criterion) {
    let mut group = c.benchmark_group("optics");

    for size in [50, 100, 200].iter() {
        let data = generate_clustered_data(*size, 10, 5);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_| {
            let optics = OPTICS::new(2.0, 5);

            b.iter(|| {
                optics.fit(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

// ================================================================================================
// Online K-Means Benchmarks (Streaming)
// ================================================================================================

fn bench_online_kmeans_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_kmeans_batch");

    for batch_size in [10, 50, 100].iter() {
        let data = generate_clustered_data(*batch_size, 10, 5);

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &_| {
                b.iter_batched(
                    || OnlineKMeans::new(5).unwrap(),
                    |mut online_kmeans| {
                        online_kmeans.update_batch(black_box(&data)).unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_online_kmeans_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_kmeans_single");

    // Test single point updates
    let point = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]).unwrap();

    group.bench_function("single_point_update", |b| {
        let mut online_kmeans = OnlineKMeans::new(3).unwrap();
        // Pre-initialize with some data
        let init_data = generate_clustered_data(50, 5, 3);
        online_kmeans.update_batch(&init_data).unwrap();

        b.iter(|| {
            online_kmeans.update_single(black_box(&point)).unwrap();
        });
    });

    group.finish();
}

// ================================================================================================
// Algorithm Comparison Benchmark
// ================================================================================================

fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");
    let data = generate_clustered_data(500, 10, 5);

    group.bench_function("kmeans_lloyd", |b| {
        let kmeans = KMeans::new(5)
            .algorithm(KMeansAlgorithm::Lloyd)
            .max_iters(50)
            .random_state(42);
        b.iter(|| {
            kmeans.fit(black_box(&data)).unwrap();
        });
    });

    group.bench_function("kmeans_elkan", |b| {
        let kmeans = KMeans::new(5)
            .algorithm(KMeansAlgorithm::Elkan)
            .max_iters(50)
            .random_state(42);
        b.iter(|| {
            kmeans.fit(black_box(&data)).unwrap();
        });
    });

    group.bench_function("kmeans_minibatch", |b| {
        let kmeans = KMeans::new(5)
            .algorithm(KMeansAlgorithm::MiniBatch)
            .max_iters(50)
            .random_state(42);
        b.iter(|| {
            kmeans.fit(black_box(&data)).unwrap();
        });
    });

    group.bench_function("gmm_diagonal", |b| {
        let gmm = GaussianMixture::new(5)
            .covariance_type(CovarianceType::Diag)
            .max_iters(50)
            .random_state(42);
        b.iter(|| {
            gmm.fit(black_box(&data)).unwrap();
        });
    });

    group.bench_function("hierarchical_average", |b| {
        let hierarchical = AgglomerativeClustering::new(5).linkage(Linkage::Average);
        b.iter(|| {
            hierarchical.fit(black_box(&data)).unwrap();
        });
    });

    group.finish();
}

criterion_group!(density_benches, bench_dbscan, bench_hdbscan, bench_optics,);

criterion_group!(
    hierarchical_benches,
    bench_hierarchical,
    bench_hierarchical_linkages,
);

criterion_group!(spectral_benches, bench_spectral,);

criterion_group!(distance_benches, bench_distance_computations,);

criterion_group!(
    online_benches,
    bench_online_kmeans_batch,
    bench_online_kmeans_single,
);

criterion_group!(comparison_benches, bench_algorithm_comparison,);

criterion_main!(
    kmeans_benches,
    gmm_benches,
    density_benches,
    hierarchical_benches,
    spectral_benches,
    distance_benches,
    online_benches,
    comparison_benches,
);
