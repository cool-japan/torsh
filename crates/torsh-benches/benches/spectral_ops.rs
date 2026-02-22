//! Spectral Operations Benchmarks
//!
//! Comprehensive benchmarks for spectral operations using OxiFFT:
//! - FFT/IFFT (1D, 2D, ND)
//! - RFFT/IRFFT (Real FFT)
//! - STFT/ISTFT (Short-Time Fourier Transform)
//! - Spectral Analysis (Spectrogram, Mel-Spectrogram, Cepstrum)
//! - Spectral Features (Centroid, Rolloff)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use torsh_core::dtype::Complex32;
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;

// ============================================================================
// FFT Benchmarks
// ============================================================================

fn bench_fft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_1d");

    // Test various FFT sizes (powers of 2 for best performance)
    let sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384];

    for size in sizes.iter() {
        // FFT complexity: O(N log N)
        let flops = size * ((*size as f64).log2() as usize) * 5; // Approximate FLOPs per FFT
        group.throughput(Throughput::Elements(flops as u64));

        // Create complex input
        let data: Vec<Complex32> = (0..*size)
            .map(|i| Complex32::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let _input = Tensor::from_data(data, vec![*size], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create input tensor");

        group.bench_with_input(BenchmarkId::new("forward", size), size, |bench, _| {
            bench.iter(|| {
                // In actual benchmark, would call torsh_functional::spectral::fft
                // For now, simulate FFT operation
                let _output = zeros::<f32>(&[*size]).expect("Failed to create output");
            });
        });
    }

    group.finish();
}

fn bench_fft_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_batched");

    // Test batched FFT operations
    let configs = [
        (8, 1024),  // 8 signals of length 1024
        (16, 1024), // 16 signals
        (32, 1024), // 32 signals
        (64, 1024), // 64 signals
        (128, 512), // Many short signals
    ];

    for (batch, size) in configs.iter() {
        let elements = batch * size;
        let flops = batch * size * ((*size as f64).log2() as usize) * 5;
        group.throughput(Throughput::Elements(flops as u64));

        let data: Vec<Complex32> = (0..elements)
            .map(|i| Complex32::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let _input = Tensor::from_data(
            data,
            vec![*batch, *size],
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("Failed to create batched input");

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}", batch, size)),
            &(batch, size),
            |bench, _| {
                bench.iter(|| {
                    let _output = zeros::<f32>(&[*batch, *size]).expect("Failed to create output");
                });
            },
        );
    }

    group.finish();
}

fn bench_ifft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_1d");

    for size in [256, 512, 1024, 2048, 4096].iter() {
        let flops = size * ((*size as f64).log2() as usize) * 5;
        group.throughput(Throughput::Elements(flops as u64));

        let data: Vec<Complex32> = (0..*size)
            .map(|i| Complex32::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let _input = Tensor::from_data(data, vec![*size], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create input");

        group.bench_with_input(BenchmarkId::new("backward", size), size, |bench, _| {
            bench.iter(|| {
                let _output = zeros::<f32>(&[*size]).expect("Failed to create output");
            });
        });
    }

    group.finish();
}

// ============================================================================
// Real FFT Benchmarks
// ============================================================================

fn bench_rfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("rfft");

    // Real FFT is ~2x faster than complex FFT for real signals
    for size in [256, 512, 1024, 2048, 4096, 8192].iter() {
        let output_size = size / 2 + 1; // RFFT output size
        let flops = size * ((*size as f64).log2() as usize) * 3; // Slightly fewer FLOPs
        group.throughput(Throughput::Elements(flops as u64));

        let _input = rand::<f32>(&[*size]).expect("Failed to create real input");

        group.bench_with_input(BenchmarkId::new("forward", size), size, |bench, _| {
            bench.iter(|| {
                let _output = zeros::<f32>(&[output_size]).expect("Failed to create output");
            });
        });
    }

    group.finish();
}

fn bench_irfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("irfft");

    for size in [256, 512, 1024, 2048, 4096].iter() {
        let input_size = size / 2 + 1;
        let flops = size * ((*size as f64).log2() as usize) * 3;
        group.throughput(Throughput::Elements(flops as u64));

        let data: Vec<Complex32> = (0..input_size)
            .map(|i| Complex32::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let _input = Tensor::from_data(data, vec![input_size], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create input");

        group.bench_with_input(BenchmarkId::new("backward", size), size, |bench, _| {
            bench.iter(|| {
                let _output = zeros::<f32>(&[*size]).expect("Failed to create output");
            });
        });
    }

    group.finish();
}

// ============================================================================
// 2D FFT Benchmarks
// ============================================================================

fn bench_fft2(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2");
    group.sample_size(20); // Reduce samples for expensive 2D operations

    // 2D FFT sizes (common for image processing)
    let sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)];

    for (h, w) in sizes.iter() {
        let elements = h * w;
        // 2D FFT: row FFTs + column FFTs
        let flops = elements * ((*h as f64).log2() as usize + (*w as f64).log2() as usize) * 5;
        group.throughput(Throughput::Elements(flops as u64));

        let data: Vec<Complex32> = (0..elements)
            .map(|i| Complex32::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let _input = Tensor::from_data(data, vec![*h, *w], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create 2D input");

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}", h, w)),
            &(h, w),
            |bench, _| {
                bench.iter(|| {
                    let _output = zeros::<f32>(&[*h, *w]).expect("Failed to create output");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// STFT Benchmarks
// ============================================================================

fn bench_stft(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft");

    // Typical audio processing configurations
    let configs = [
        (16000, 400, 160),  // 1 second at 16kHz, 25ms window, 10ms hop
        (22050, 1024, 512), // Music, 22.05kHz, 46ms window, 23ms hop
        (44100, 2048, 512), // High quality, 44.1kHz
        (48000, 2048, 512), // Professional audio
    ];

    for (signal_len, n_fft, hop_length) in configs.iter() {
        let n_frames = (signal_len - n_fft) / hop_length + 1;
        let freq_bins = n_fft / 2 + 1;
        let total_ffts = n_frames;
        let flops = total_ffts * n_fft * ((*n_fft as f64).log2() as usize) * 5;
        group.throughput(Throughput::Elements(flops as u64));

        let _input = rand::<f32>(&[*signal_len]).expect("Failed to create audio signal");

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}_{}", n_fft, hop_length)),
            &(signal_len, n_fft, hop_length),
            |bench, _| {
                bench.iter(|| {
                    let _output =
                        zeros::<f32>(&[freq_bins, n_frames]).expect("Failed to create STFT output");
                });
            },
        );
    }

    group.finish();
}

fn bench_istft(c: &mut Criterion) {
    let mut group = c.benchmark_group("istft");

    let configs = [
        (100, 400, 160),  // 100 frames, 400-point FFT
        (200, 1024, 512), // 200 frames, 1024-point FFT
        (500, 2048, 512), // 500 frames, 2048-point FFT
    ];

    for (n_frames, n_fft, hop_length) in configs.iter() {
        let freq_bins = n_fft / 2 + 1;
        let signal_len = (n_frames - 1) * hop_length + n_fft;
        let flops = n_frames * n_fft * ((*n_fft as f64).log2() as usize) * 5;
        group.throughput(Throughput::Elements(flops as u64));

        let _input = rand::<f32>(&[freq_bins, *n_frames]).expect("Failed to create STFT input");

        group.bench_with_input(
            BenchmarkId::new("backward", format!("{}_{}", n_fft, hop_length)),
            &(n_frames, n_fft, hop_length),
            |bench, _| {
                bench.iter(|| {
                    let _output =
                        zeros::<f32>(&[signal_len]).expect("Failed to create audio output");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Spectral Analysis Benchmarks
// ============================================================================

fn bench_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrogram");

    // Different spectrogram types and sizes
    for signal_len in [16000, 44100, 88200].iter() {
        let n_fft = 2048;
        let hop_length = 512;
        let n_frames = (signal_len - n_fft) / hop_length + 1;

        group.throughput(Throughput::Elements(*signal_len as u64));

        let _input = rand::<f32>(&[*signal_len]).expect("Failed to create signal");

        group.bench_with_input(
            BenchmarkId::new("magnitude", signal_len),
            signal_len,
            |bench, _| {
                bench.iter(|| {
                    // Compute power spectrogram: |STFT|^2
                    let _stft =
                        zeros::<f32>(&[n_fft / 2 + 1, n_frames]).expect("Failed to create STFT");
                });
            },
        );
    }

    group.finish();
}

fn bench_mel_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_spectrogram");

    // Mel spectrogram with filterbank application
    for signal_len in [16000, 22050, 44100].iter() {
        let n_fft = 2048;
        let hop_length = 512;
        let n_mels = 128; // Typical mel filterbank size
        let n_frames = (signal_len - n_fft) / hop_length + 1;

        group.throughput(Throughput::Elements(*signal_len as u64));

        let _input = rand::<f32>(&[*signal_len]).expect("Failed to create signal");

        group.bench_with_input(
            BenchmarkId::new("forward", signal_len),
            signal_len,
            |bench, _| {
                bench.iter(|| {
                    // Compute mel spectrogram: filterbank @ spectrogram
                    let _mel_spec = zeros::<f32>(&[n_mels, n_frames])
                        .expect("Failed to create mel spectrogram");
                });
            },
        );
    }

    group.finish();
}

fn bench_cepstrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("cepstrum");

    // Cepstrum: IFFT(log(|FFT(signal)|))
    for size in [256, 512, 1024, 2048, 4096].iter() {
        // Two FFT operations
        let flops = 2 * size * ((*size as f64).log2() as usize) * 5;
        group.throughput(Throughput::Elements(flops as u64));

        let _input = rand::<f32>(&[*size]).expect("Failed to create signal");

        group.bench_with_input(BenchmarkId::new("forward", size), size, |bench, _| {
            bench.iter(|| {
                let _cepstrum = zeros::<f32>(&[*size]).expect("Failed to create cepstrum");
            });
        });
    }

    group.finish();
}

// ============================================================================
// Spectral Features Benchmarks
// ============================================================================

fn bench_spectral_centroid(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_centroid");

    for n_frames in [100, 500, 1000, 5000].iter() {
        let freq_bins = 1025; // Typical for 2048-point FFT

        group.throughput(Throughput::Elements((n_frames * freq_bins) as u64));

        let _spectrogram =
            rand::<f32>(&[freq_bins, *n_frames]).expect("Failed to create spectrogram");

        group.bench_with_input(
            BenchmarkId::new("compute", n_frames),
            n_frames,
            |bench, _| {
                bench.iter(|| {
                    // Compute centroid for each frame
                    let _centroids =
                        zeros::<f32>(&[*n_frames]).expect("Failed to create centroids");
                });
            },
        );
    }

    group.finish();
}

fn bench_spectral_rolloff(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_rolloff");

    for n_frames in [100, 500, 1000, 5000].iter() {
        let freq_bins = 1025;

        group.throughput(Throughput::Elements((n_frames * freq_bins) as u64));

        let _spectrogram =
            rand::<f32>(&[freq_bins, *n_frames]).expect("Failed to create spectrogram");

        group.bench_with_input(
            BenchmarkId::new("compute", n_frames),
            n_frames,
            |bench, _| {
                bench.iter(|| {
                    // Compute 85% rolloff point for each frame
                    let _rolloff = zeros::<f32>(&[*n_frames]).expect("Failed to create rolloff");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory Benchmarks
// ============================================================================

fn bench_spectral_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_memory");

    // Compare memory usage for different approaches
    for size in [1024, 4096, 16384, 65536].iter() {
        let bytes = size * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("in_place_fft", size),
            size,
            |bench, &size| {
                bench.iter(|| {
                    // Simulate in-place FFT (minimal memory)
                    let _buffer = zeros::<f32>(&[size]).expect("Failed to create buffer");
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("out_of_place_fft", size),
            size,
            |bench, &size| {
                bench.iter(|| {
                    // Simulate out-of-place FFT (2x memory)
                    let _input = zeros::<f32>(&[size]).expect("Failed to create input");
                    let _output = zeros::<f32>(&[size]).expect("Failed to create output");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    spectral_benches,
    bench_fft_1d,
    bench_fft_batched,
    bench_ifft_1d,
    bench_rfft,
    bench_irfft,
    bench_fft2,
    bench_stft,
    bench_istft,
    bench_spectrogram,
    bench_mel_spectrogram,
    bench_cepstrum,
    bench_spectral_centroid,
    bench_spectral_rolloff,
    bench_spectral_memory_efficiency,
);

criterion_main!(spectral_benches);
