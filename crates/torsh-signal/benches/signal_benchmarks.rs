//! Benchmarks for torsh-signal signal processing operations
//!
//! These benchmarks measure the performance of key signal processing operations
//! to ensure optimal performance and identify bottlenecks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use torsh_signal::prelude::*;
use torsh_tensor::creation::{ones, zeros};

/// Benchmark window function generation
fn benchmark_windows(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_functions");

    // Test different window sizes
    let sizes = [64, 256, 1024, 4096, 8192];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("hamming", size), &size, |b, &size| {
            b.iter(|| hamming_window(black_box(size), false))
        });

        group.bench_with_input(BenchmarkId::new("hann", size), &size, |b, &size| {
            b.iter(|| hann_window(black_box(size), false))
        });

        group.bench_with_input(BenchmarkId::new("blackman", size), &size, |b, &size| {
            b.iter(|| blackman_window(black_box(size), false))
        });

        group.bench_with_input(BenchmarkId::new("kaiser", size), &size, |b, &size| {
            b.iter(|| kaiser_window(black_box(size), 5.0, false))
        });

        group.bench_with_input(BenchmarkId::new("gaussian", size), &size, |b, &size| {
            b.iter(|| gaussian_window(black_box(size), 0.4, false))
        });

        group.bench_with_input(BenchmarkId::new("tukey", size), &size, |b, &size| {
            b.iter(|| tukey_window(black_box(size), 0.5, false))
        });
    }

    group.finish();
}

/// Benchmark STFT operations
fn benchmark_stft(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_operations");

    // Different signal lengths and parameters
    let configs = [
        (1024, 256, 128), // (signal_len, n_fft, hop_length)
        (4096, 512, 256),
        (8192, 1024, 512),
        (16384, 2048, 1024),
    ];

    for (signal_len, n_fft, hop_length) in configs {
        let signal = ones(&[signal_len]).unwrap();

        group.throughput(Throughput::Elements(signal_len as u64));

        let params = StftParams {
            n_fft,
            hop_length: Some(hop_length),
            win_length: Some(n_fft),
            window: Some(Window::Hann),
            center: true,
            normalized: false,
            onesided: true,
            return_complex: true,
        };

        group.bench_with_input(
            BenchmarkId::new("stft", format!("{}_{}", signal_len, n_fft)),
            &(&signal, &params),
            |b, (signal, params)| b.iter(|| stft(black_box(signal), black_box(**params))),
        );
    }

    group.finish();
}

/// Benchmark spectrogram computation
fn benchmark_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrogram_operations");

    let signal_lengths = [2048, 8192, 16384, 32768];

    for signal_len in signal_lengths {
        let signal = ones(&[signal_len]).unwrap();

        group.throughput(Throughput::Elements(signal_len as u64));

        group.bench_with_input(
            BenchmarkId::new("spectrogram", signal_len),
            &signal,
            |b, signal| {
                b.iter(|| {
                    spectrogram(
                        black_box(signal),
                        1024,               // n_fft
                        Some(512),          // hop_length
                        Some(1024),         // win_length
                        Some(Window::Hann), // window
                        true,               // center
                        "reflect",          // pad_mode
                        false,              // normalized
                        true,               // onesided
                        Some(2.0),          // power
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark mel-scale operations
fn benchmark_mel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_operations");

    // Create test spectrograms of different sizes
    let configs = [
        (257, 100, 80), // (n_freqs, n_frames, n_mels)
        (513, 200, 128),
        (1025, 300, 128),
        (2049, 500, 256),
    ];

    for (n_freqs, n_frames, n_mels) in configs {
        let specgram = ones(&[n_freqs, n_frames]).unwrap();
        let sample_rate = 16000.0;
        let f_min = 0.0;
        let f_max = 8000.0;

        group.throughput(Throughput::Elements((n_freqs * n_frames) as u64));

        // Benchmark mel filterbank creation
        group.bench_with_input(
            BenchmarkId::new("create_fb_matrix", format!("{}_{}", n_freqs, n_mels)),
            &(n_freqs, n_mels),
            |b, &(n_freqs, n_mels)| {
                b.iter(|| {
                    create_fb_matrix(
                        black_box(n_freqs),
                        black_box(f_min),
                        black_box(f_max),
                        black_box(n_mels),
                        black_box(sample_rate),
                    )
                })
            },
        );

        // Benchmark mel spectrogram computation
        group.bench_with_input(
            BenchmarkId::new("mel_spectrogram", format!("{}x{}", n_freqs, n_frames)),
            &specgram,
            |b, specgram| {
                b.iter(|| {
                    mel_spectrogram(
                        black_box(specgram),
                        black_box(f_min),
                        Some(black_box(f_max)),
                        black_box(n_mels),
                        black_box(sample_rate),
                    )
                })
            },
        );

        // Benchmark mel scale conversion
        let freq_tensor = ones(&[n_freqs]).unwrap();
        group.bench_with_input(
            BenchmarkId::new("mel_scale", n_freqs),
            &freq_tensor,
            |b, freq_tensor| {
                b.iter(|| {
                    mel_scale(
                        black_box(freq_tensor),
                        black_box(f_min),
                        Some(black_box(f_max)),
                        black_box(n_mels),
                        black_box(sample_rate),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark filter operations
fn benchmark_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_operations");

    let signal_lengths = [1024, 4096, 8192, 16384];
    let sample_rate = 8000.0;

    for signal_len in signal_lengths {
        let signal = ones(&[signal_len]).unwrap();

        group.throughput(Throughput::Elements(signal_len as u64));

        // Benchmark different filter types
        group.bench_with_input(
            BenchmarkId::new("lowpass", signal_len),
            &signal,
            |b, signal| {
                b.iter(|| {
                    lowpass_filter(
                        black_box(signal),
                        black_box(1000.0),
                        black_box(sample_rate),
                        black_box(4),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bandpass", signal_len),
            &signal,
            |b, signal| {
                b.iter(|| {
                    bandpass_filter(
                        black_box(signal),
                        black_box(500.0),
                        black_box(2000.0),
                        black_box(sample_rate),
                        black_box(4),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("median_filter", signal_len),
            &signal,
            |b, signal| b.iter(|| median_filter(black_box(signal), black_box(5))),
        );

        group.bench_with_input(
            BenchmarkId::new("gaussian_filter", signal_len),
            &signal,
            |b, signal| b.iter(|| gaussian_filter(black_box(signal), black_box(1.5))),
        );
    }

    group.finish();
}

/// Benchmark convolution operations
fn benchmark_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolution_operations");

    let configs = [
        (1024, 8), // (signal_len, kernel_len)
        (2048, 16),
        (4096, 32),
        (8192, 64),
    ];

    for (signal_len, kernel_len) in configs {
        let signal = ones(&[signal_len]).unwrap();
        let kernel = ones(&[kernel_len]).unwrap();

        group.throughput(Throughput::Elements(signal_len as u64));

        // Test different convolution modes
        let modes = ["full", "valid", "same"];

        for mode in modes {
            group.bench_with_input(
                BenchmarkId::new(
                    "convolve1d",
                    format!("{}_{}_{}", signal_len, kernel_len, mode),
                ),
                &(&signal, &kernel, mode),
                |b, (signal, kernel, mode)| {
                    b.iter(|| {
                        convolve1d(
                            black_box(signal),
                            black_box(kernel),
                            black_box(mode),
                            "auto",
                        )
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    "correlate1d",
                    format!("{}_{}_{}", signal_len, kernel_len, mode),
                ),
                &(&signal, &kernel, mode),
                |b, (signal, kernel, mode)| {
                    b.iter(|| correlate1d(black_box(signal), black_box(kernel), black_box(mode)))
                },
            );
        }
    }

    group.finish();
}

/// Benchmark window normalization
fn benchmark_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_normalization");

    let sizes = [256, 1024, 4096, 8192];

    for size in sizes {
        let window = hamming_window(size, false).unwrap();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("magnitude_norm", size),
            &window,
            |b, window| b.iter(|| normalize_window(black_box(window), "magnitude")),
        );

        group.bench_with_input(
            BenchmarkId::new("power_norm", size),
            &window,
            |b, window| b.iter(|| normalize_window(black_box(window), "power")),
        );
    }

    group.finish();
}

/// Benchmark memory-intensive operations
fn benchmark_memory_intensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_intensive");

    // Large signal processing
    let large_signal_len = 65536; // 64k samples
    let signal = ones(&[large_signal_len]).unwrap();

    group.throughput(Throughput::Elements(large_signal_len as u64));

    // Large STFT
    let large_stft_params = StftParams {
        n_fft: 2048,
        hop_length: Some(1024),
        win_length: Some(2048),
        window: Some(Window::Hann),
        center: true,
        normalized: false,
        onesided: true,
        return_complex: true,
    };

    group.bench_function("large_stft", |b| {
        b.iter(|| stft(black_box(&signal), black_box(large_stft_params.clone())))
    });

    // Large spectrogram
    group.bench_function("large_spectrogram", |b| {
        b.iter(|| {
            spectrogram(
                black_box(&signal),
                2048,
                Some(1024),
                Some(2048),
                Some(Window::Hann),
                true,
                "reflect",
                false,
                true,
                Some(2.0),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_windows,
    benchmark_stft,
    benchmark_spectrogram,
    benchmark_mel_operations,
    benchmark_filters,
    benchmark_convolution,
    benchmark_normalization,
    benchmark_memory_intensive
);

criterion_main!(benches);
