//! Integration tests for torsh-signal
//!
//! These tests verify that different modules work together correctly
//! and that the overall API provides a cohesive signal processing experience.

use approx::assert_relative_eq;
use torsh_core::error::Result;
use torsh_signal::prelude::*;
use torsh_tensor::creation::{ones, zeros};

/// Test complete signal processing pipeline: window -> STFT -> spectrogram
#[test]
fn test_signal_processing_pipeline() -> Result<()> {
    // Create a test signal
    let signal_length = 1024;
    let signal = ones(&[signal_length])?;

    // Apply window function
    let window = hamming_window(512, false)?;
    assert_eq!(window.shape().dims(), &[512]);

    // Compute STFT
    let stft_params = StftParams {
        n_fft: 512,
        hop_length: Some(256),
        win_length: Some(512),
        window: Some(Window::Hamming),
        center: true,
        normalized: false,
        onesided: true,
        return_complex: true,
    };

    let stft_result = stft(&signal, stft_params)?;
    assert_eq!(stft_result.shape().ndim(), 2); // Should be 2D: [frequencies, time_frames]

    // Compute spectrogram
    let spec = spectrogram(
        &signal,
        512,       // n_fft
        Some(256), // hop_length
        Some(512), // win_length
        Some(Window::Hamming),
        true,      // center
        "reflect", // pad_mode
        false,     // normalized
        true,      // onesided
        Some(2.0), // power
    )?;

    assert_eq!(spec.shape().ndim(), 2);
    assert_eq!(spec.shape().dims()[0], 257); // n_fft/2 + 1 for onesided

    Ok(())
}

/// Test mel-scale signal processing pipeline
#[test]
fn test_mel_scale_pipeline() -> Result<()> {
    let signal_length = 1024;
    let signal = ones(&[signal_length])?;

    // Compute regular spectrogram first
    let spec = spectrogram(
        &signal,
        512,
        Some(256),
        Some(512),
        Some(Window::Hann),
        true,
        "reflect",
        false,
        true,
        Some(2.0),
    )?;

    // Convert to mel scale
    let n_mels = 80;
    let sample_rate = 16000.0;
    let f_min = 0.0;
    let f_max = Some(8000.0);

    let mel_spec = mel_spectrogram(&spec, f_min, f_max, n_mels, sample_rate)?;
    assert_eq!(mel_spec.shape().dims()[0], n_mels);

    // Test mel scale conversion functions
    let n_freqs = spec.shape().dims()[0]; // Should be 257 for n_fft=512 onesided
    let freq_tensor = ones(&[n_freqs])?;

    let mel_result = mel_scale(&freq_tensor, f_min, f_max, n_mels, sample_rate)?;
    assert_eq!(mel_result.shape().dims()[0], n_mels);

    // Test inverse mel scale
    let inverse_result = inverse_mel_scale(&mel_result, f_min, f_max, n_freqs, sample_rate)?;
    assert_eq!(inverse_result.shape().dims()[0], n_freqs);

    Ok(())
}

/// Test window function combinations and properties
#[test]
fn test_window_combinations() -> Result<()> {
    let n = 256;

    // Test different window types
    let windows = vec![
        Window::Rectangular,
        Window::Hamming,
        Window::Hann,
        Window::Blackman,
        Window::Bartlett,
        Window::Kaiser(5.0),
        Window::Gaussian(0.4),
        Window::Tukey(0.5),
        Window::Cosine,
    ];

    for window_type in windows {
        let w = window(window_type, n, false)?;
        assert_eq!(w.shape().dims(), &[n]);

        // Test normalization
        let normalized = normalize_window(&w, "magnitude")?;
        let sum = normalized.sum()?;
        assert_relative_eq!(sum.item().unwrap(), 1.0, epsilon = 1e-4);

        // Test power normalization
        let power_normalized = normalize_window(&w, "power")?;
        let power_sum = power_normalized.pow_scalar(2.0)?.sum()?;
        assert_relative_eq!(power_sum.item().unwrap(), 1.0, epsilon = 1e-4);
    }

    Ok(())
}

/// Test filter combinations and frequency response
#[test]
fn test_filter_combinations() -> Result<()> {
    let signal = ones(&[1000])?;
    let sample_rate = 8000.0;

    // Test different filter types
    let lowpass = lowpass_filter(&signal, 1000.0, sample_rate, 4)?;
    assert_eq!(lowpass.shape().dims(), signal.shape().dims());

    let highpass = highpass_filter(&signal, 1000.0, sample_rate, 4)?;
    assert_eq!(highpass.shape().dims(), signal.shape().dims());

    let bandpass = bandpass_filter(&signal, 500.0, 2000.0, sample_rate, 4)?;
    assert_eq!(bandpass.shape().dims(), signal.shape().dims());

    let bandstop = bandstop_filter(&signal, 500.0, 2000.0, sample_rate, 4)?;
    assert_eq!(bandstop.shape().dims(), signal.shape().dims());

    // Test filter design functions
    let (num, den) = butterworth_lowpass(1000.0, sample_rate, 4)?;
    assert_eq!(num.shape().dims(), &[5]); // order + 1
    assert_eq!(den.shape().dims(), &[5]);

    // Test frequency response
    let response = freqz(&num, &den, 512, sample_rate)?;
    assert_eq!(response.frequencies.shape().dims(), &[512]);
    assert_eq!(response.magnitude.shape().dims(), &[512]);
    assert_eq!(response.phase.shape().dims(), &[512]);

    Ok(())
}

/// Test STFT/ISTFT round-trip consistency
#[test]
fn test_stft_istft_roundtrip() -> Result<()> {
    let signal = ones(&[1024])?;
    let n_fft = 256;
    let hop_length = Some(128);
    let win_length = Some(256);
    let window = Some(Window::Hann);

    // Forward STFT
    let stft_params = StftParams {
        n_fft,
        hop_length,
        win_length,
        window,
        center: true,
        normalized: false,
        onesided: true,
        return_complex: true,
    };

    let stft_result = stft(&signal, stft_params)?;
    assert_eq!(stft_result.shape().ndim(), 2);

    // Inverse STFT
    let reconstructed = istft(
        &stft_result,
        n_fft,
        hop_length,
        win_length,
        window,
        true,                           // center
        false,                          // normalized
        true,                           // onesided
        Some(signal.shape().dims()[0]), // length
    )?;

    // Should have same length as original (or close due to windowing)
    assert!(reconstructed.shape().dims()[0] >= signal.shape().dims()[0] - n_fft);

    Ok(())
}

/// Test error handling across modules
#[test]
fn test_error_handling() {
    // Window errors
    assert!(normalize_window(&ones(&[10]).unwrap(), "invalid").is_err());

    // Filter errors
    let signal_2d = ones(&[10, 10]).unwrap();
    assert!(lowpass_filter(&signal_2d, 1000.0, 8000.0, 4).is_err());
    assert!(median_filter(&ones(&[10]).unwrap(), 6).is_err()); // Even window
    assert!(gaussian_filter(&ones(&[10]).unwrap(), -1.0).is_err()); // Negative sigma

    // Convolution errors
    let big_kernel = ones(&[20]).unwrap();
    let small_signal = ones(&[10]).unwrap();
    assert!(convolve1d(&small_signal, &big_kernel, "full", "auto").is_err());
    assert!(convolve1d(&small_signal, &ones(&[3]).unwrap(), "invalid", "auto").is_err());

    // Filter parameter errors
    assert!(bandpass_filter(&ones(&[100]).unwrap(), 2000.0, 1000.0, 8000.0, 4).is_err());
    assert!(savgol_filter(&ones(&[100]).unwrap(), 6, 2).is_err()); // Even window
    assert!(savgol_filter(&ones(&[100]).unwrap(), 5, 5).is_err()); // Order >= window

    // STFT errors
    let invalid_tensor = ones(&[10, 10, 10]).unwrap(); // 3D tensor
    let stft_params = StftParams::default();
    assert!(stft(&invalid_tensor, stft_params).is_err());
}

/// Test performance and memory efficiency with larger signals
#[test]
#[ignore]
fn test_performance_with_large_signals() -> Result<()> {
    // Create a larger signal to test performance characteristics
    let signal_length = 16384; // 16k samples
    let signal = ones(&[signal_length])?;

    // Test STFT with reasonable parameters
    let stft_params = StftParams {
        n_fft: 1024,
        hop_length: Some(512),
        win_length: Some(1024),
        window: Some(Window::Hann),
        center: true,
        normalized: false,
        onesided: true,
        return_complex: true,
    };

    let stft_result = stft(&signal, stft_params)?;

    // Verify output dimensions are reasonable
    assert_eq!(stft_result.shape().ndim(), 2);
    let n_freqs = stft_result.shape().dims()[0];
    let n_frames = stft_result.shape().dims()[1];

    assert_eq!(n_freqs, 513); // 1024/2 + 1 for onesided
    assert!(n_frames > 0);
    assert!(n_frames < signal_length); // Should be much smaller than input

    // Test spectrogram computation
    let spec = spectrogram(
        &signal,
        1024,
        Some(512),
        Some(1024),
        Some(Window::Hann),
        true,
        "reflect",
        false,
        true,
        Some(2.0),
    )?;

    assert_eq!(spec.shape().dims()[0], n_freqs);
    assert_eq!(spec.shape().dims()[1], n_frames);

    Ok(())
}

/// Test consistency between different mel-scale approaches
#[test]
fn test_mel_scale_consistency() -> Result<()> {
    let n_freqs = 513;
    let n_mels = 80;
    let sample_rate = 16000.0;
    let f_min = 0.0;
    let f_max = 8000.0;

    // Create frequency-domain data
    let specgram = ones(&[n_freqs, 100])?; // 100 time frames

    // Method 1: Using mel_spectrogram
    let mel_spec1 = mel_spectrogram(&specgram, f_min, Some(f_max), n_mels, sample_rate)?;

    // Method 2: Using create_fb_matrix manually
    let fb_matrix = create_fb_matrix(n_freqs, f_min, f_max, n_mels, sample_rate)?;
    let mel_spec2 = fb_matrix.matmul(&specgram)?;

    // Results should be identical
    assert_eq!(mel_spec1.shape().dims(), mel_spec2.shape().dims());
    assert_eq!(mel_spec1.shape().dims(), &[n_mels, 100]);

    // Values should be close (allowing for numerical differences)
    let diff = mel_spec1.sub(&mel_spec2)?.abs()?.sum()?;
    assert!(diff.item().unwrap() < 1e-5);

    Ok(())
}
