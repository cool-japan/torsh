//! Performance benchmarks for torsh-metrics
//!
//! This module provides comprehensive benchmarks for all metrics,
//! allowing comparison with reference implementations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use torsh_core::device::DeviceType;
use torsh_metrics::{
    classification::{Accuracy, F1Score, Precision, Recall},
    parallel::ParallelAccuracy,
    regression::{R2Score, MAE, MSE, RMSE},
    streaming::StreamingAccuracy,
    Metric,
};
use torsh_tensor::creation::from_vec;

fn create_binary_classification_data(
    n_samples: usize,
) -> (torsh_tensor::Tensor, torsh_tensor::Tensor) {
    let mut predictions = Vec::with_capacity(n_samples * 2);
    let mut targets = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let target = (i % 2) as f32;
        targets.push(target);

        if target == 0.0 {
            predictions.push(0.7);
            predictions.push(0.3);
        } else {
            predictions.push(0.3);
            predictions.push(0.7);
        }
    }

    let preds = from_vec(predictions, &[n_samples, 2], DeviceType::Cpu).unwrap();
    let targs = from_vec(targets, &[n_samples], DeviceType::Cpu).unwrap();

    (preds, targs)
}

fn create_multiclass_data(
    n_samples: usize,
    n_classes: usize,
) -> (torsh_tensor::Tensor, torsh_tensor::Tensor) {
    let mut predictions = Vec::with_capacity(n_samples * n_classes);
    let mut targets = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let target = (i % n_classes) as f32;
        targets.push(target);

        for j in 0..n_classes {
            if j == (i % n_classes) {
                predictions.push(0.8);
            } else {
                predictions.push(0.2 / (n_classes - 1) as f32);
            }
        }
    }

    let preds = from_vec(predictions, &[n_samples, n_classes], DeviceType::Cpu).unwrap();
    let targs = from_vec(targets, &[n_samples], DeviceType::Cpu).unwrap();

    (preds, targs)
}

fn create_regression_data(n_samples: usize) -> (torsh_tensor::Tensor, torsh_tensor::Tensor) {
    let mut predictions = Vec::with_capacity(n_samples);
    let mut targets = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x = i as f32 * 0.1;
        targets.push(x * 2.0 + 1.0);
        predictions.push(x * 2.0 + 1.0 + (i as f32 % 3.0) * 0.1);
    }

    let preds = from_vec(predictions, &[n_samples], DeviceType::Cpu).unwrap();
    let targs = from_vec(targets, &[n_samples], DeviceType::Cpu).unwrap();

    (preds, targs)
}

fn benchmark_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let (predictions, targets) = create_binary_classification_data(*size);

        group.bench_with_input(BenchmarkId::new("standard", size), size, |b, _| {
            let accuracy = Accuracy::new();
            b.iter(|| {
                black_box(accuracy.compute(&predictions, &targets));
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel", size), size, |b, _| {
            let accuracy = ParallelAccuracy::new();
            b.iter(|| {
                black_box(accuracy.compute(&predictions, &targets));
            });
        });

        group.bench_with_input(BenchmarkId::new("streaming", size), size, |b, _| {
            b.iter(|| {
                let mut accuracy = StreamingAccuracy::new();
                accuracy.update(&predictions, &targets);
                black_box(accuracy.compute());
            });
        });
    }

    group.finish();
}

fn benchmark_precision_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_recall");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let (predictions, targets) = create_binary_classification_data(*size);

        group.bench_with_input(BenchmarkId::new("precision", size), size, |b, _| {
            let precision = Precision::micro();
            b.iter(|| {
                black_box(precision.compute(&predictions, &targets));
            });
        });

        group.bench_with_input(BenchmarkId::new("recall", size), size, |b, _| {
            let recall = Recall::micro();
            b.iter(|| {
                black_box(recall.compute(&predictions, &targets));
            });
        });

        group.bench_with_input(BenchmarkId::new("f1", size), size, |b, _| {
            let f1 = F1Score::micro();
            b.iter(|| {
                black_box(f1.compute(&predictions, &targets));
            });
        });
    }

    group.finish();
}

fn benchmark_multiclass_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiclass_accuracy");

    for n_classes in [5, 10, 100].iter() {
        let n_samples = 10_000;
        group.throughput(Throughput::Elements(n_samples as u64));

        let (predictions, targets) = create_multiclass_data(n_samples, *n_classes);

        group.bench_with_input(BenchmarkId::new("classes", n_classes), n_classes, |b, _| {
            let accuracy = Accuracy::new();
            b.iter(|| {
                black_box(accuracy.compute(&predictions, &targets));
            });
        });
    }

    group.finish();
}

fn benchmark_regression_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let (predictions, targets) = create_regression_data(*size);

        group.bench_with_input(BenchmarkId::new("mse", size), size, |b, _| {
            let mse = MSE;
            b.iter(|| {
                black_box(mse.compute(&predictions, &targets));
            });
        });

        group.bench_with_input(BenchmarkId::new("rmse", size), size, |b, _| {
            let rmse = RMSE;
            b.iter(|| {
                black_box(rmse.compute(&predictions, &targets));
            });
        });

        group.bench_with_input(BenchmarkId::new("mae", size), size, |b, _| {
            let mae = MAE;
            b.iter(|| {
                black_box(mae.compute(&predictions, &targets));
            });
        });

        group.bench_with_input(BenchmarkId::new("r2", size), size, |b, _| {
            let r2 = R2Score::new();
            b.iter(|| {
                black_box(r2.compute(&predictions, &targets));
            });
        });
    }

    group.finish();
}

fn benchmark_top_k_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k_accuracy");

    for k in [1, 5, 10].iter() {
        let n_samples = 10_000;
        let n_classes = 100;
        group.throughput(Throughput::Elements(n_samples as u64));

        let (predictions, targets) = create_multiclass_data(n_samples, n_classes);

        group.bench_with_input(BenchmarkId::new("top_k", k), k, |b, _| {
            let accuracy = Accuracy::top_k(*k);
            b.iter(|| {
                black_box(accuracy.compute(&predictions, &targets));
            });
        });
    }

    group.finish();
}

fn benchmark_batch_size_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_effects");

    for batch_size in [32, 64, 128, 256, 512, 1024].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        let (predictions, targets) = create_binary_classification_data(*batch_size);

        group.bench_with_input(
            BenchmarkId::new("accuracy", batch_size),
            batch_size,
            |b, _| {
                let accuracy = Accuracy::new();
                b.iter(|| {
                    black_box(accuracy.compute(&predictions, &targets));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_accuracy,
    benchmark_precision_recall,
    benchmark_multiclass_accuracy,
    benchmark_regression_metrics,
    benchmark_top_k_accuracy,
    benchmark_batch_size_effects
);
criterion_main!(benches);
