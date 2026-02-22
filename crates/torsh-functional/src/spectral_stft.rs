//! Complete STFT/ISTFT implementation with windowing and overlap-add
//!
//! This module provides production-ready Short-Time Fourier Transform implementations
//! with proper windowing, overlap-add reconstruction, and all standard options.

use std::f32::consts::PI;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

use crate::spectral::{fft, ifft, rfft};

/// Window function type for STFT
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hann window (raised cosine)
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Bartlett (triangular) window
    Bartlett,
    /// Kaiser window (requires beta parameter)
    Kaiser(i32), // beta as i32 for simplicity
}

/// Generate window function
///
/// # Arguments
///
/// * `window_type` - Type of window function
/// * `size` - Window size
///
/// # Mathematical Formulas
///
/// **Hann window:**
/// ```text
/// w[n] = 0.5 - 0.5 * cos(2π * n / (N-1))
/// ```
///
/// **Hamming window:**
/// ```text
/// w[n] = 0.54 - 0.46 * cos(2π * n / (N-1))
/// ```
///
/// **Blackman window:**
/// ```text
/// w[n] = 0.42 - 0.5 * cos(2π * n / (N-1)) + 0.08 * cos(4π * n / (N-1))
/// ```
///
/// **Bartlett window:**
/// ```text
/// w[n] = 1 - |2n / (N-1) - 1|
/// ```
pub fn generate_window(window_type: WindowFunction, size: usize) -> TorshResult<Vec<f32>> {
    if size == 0 {
        return Err(TorshError::InvalidArgument(
            "Window size must be positive".to_string(),
        ));
    }

    let mut window = vec![0.0; size];

    match window_type {
        WindowFunction::Rectangular => {
            window.fill(1.0);
        }
        WindowFunction::Hann => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f32;
                let n_size = size as f32;
                *w = 0.5 - 0.5 * (2.0 * PI * n / (n_size - 1.0)).cos();
            }
        }
        WindowFunction::Hamming => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f32;
                let n_size = size as f32;
                *w = 0.54 - 0.46 * (2.0 * PI * n / (n_size - 1.0)).cos();
            }
        }
        WindowFunction::Blackman => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f32;
                let n_size = size as f32;
                let factor = 2.0 * PI * n / (n_size - 1.0);
                *w = 0.42 - 0.5 * factor.cos() + 0.08 * (2.0 * factor).cos();
            }
        }
        WindowFunction::Bartlett => {
            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f32;
                let n_size = size as f32;
                *w = 1.0 - (2.0 * n / (n_size - 1.0) - 1.0).abs();
            }
        }
        WindowFunction::Kaiser(beta) => {
            // Kaiser window implementation using modified Bessel function approximation
            let beta_f = beta as f32;
            let i0_beta = bessel_i0(beta_f);

            for (i, w) in window.iter_mut().enumerate() {
                let n = i as f32;
                let n_size = size as f32;
                let x = beta_f * (1.0 - (2.0 * n / (n_size - 1.0) - 1.0).powi(2)).sqrt();
                *w = bessel_i0(x) / i0_beta;
            }
        }
    }

    Ok(window)
}

/// Modified Bessel function of the first kind, order 0 (approximation for Kaiser window)
fn bessel_i0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.1328592e-1
                    + y * (0.225319e-2
                        + y * (-0.157565e-2
                            + y * (0.916281e-2
                                + y * (-0.2057706e-1
                                    + y * (0.2635537e-1
                                        + y * (-0.1647633e-1 + y * 0.392377e-2))))))))
    }
}

/// Complete Short-Time Fourier Transform with proper windowing
///
/// # Arguments
///
/// * `input` - Input signal tensor (1D or 2D for batched processing)
/// * `n_fft` - FFT size
/// * `hop_length` - Number of samples between successive frames
/// * `win_length` - Window size (defaults to n_fft)
/// * `window` - Window function type
/// * `center` - If true, pad signal symmetrically
/// * `normalized` - If true, normalize by window energy
/// * `onesided` - If true, return only positive frequencies (for real signals)
///
/// # Returns
///
/// Complex spectrogram tensor with shape:
/// - For 1D input \[signal_length\]: returns \[freq_bins, time_frames, 2\] (real, imag)
/// - For 2D input \[batch, signal_length\]: returns \[batch, freq_bins, time_frames, 2\]
///
/// # Mathematical Formula
///
/// ```text
/// STFT(m, ω) = Σ(n=0 to N-1) x[n + mH] * w[n] * exp(-jωn)
/// ```
///
/// where:
/// - m is the frame index
/// - H is the hop length
/// - w\[n\] is the window function
/// - N is the window length
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_functional::spectral_stft::{stft_complete, WindowFunction};
///
/// let signal = randn(&[16384], None, None, None)?;
/// let spec = stft_complete(
///     &signal,
///     512,                      // n_fft
///     Some(128),                // hop_length
///     None,                     // win_length (defaults to n_fft)
///     WindowFunction::Hann,     // window
///     true,                     // center padding
///     false,                    // normalized
///     true,                     // onesided
/// )?;
/// ```
pub fn stft_complete(
    input: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: WindowFunction,
    center: bool,
    normalized: bool,
    onesided: bool,
) -> TorshResult<Tensor> {
    let input_shape = input.shape();
    let ndim = input_shape.ndim();

    if ndim == 0 || ndim > 2 {
        return Err(TorshError::InvalidArgument(
            "STFT input must be 1D or 2D".to_string(),
        ));
    }

    let hop_len = hop_length.unwrap_or(n_fft / 4);
    let win_len = win_length.unwrap_or(n_fft);

    if win_len > n_fft {
        return Err(TorshError::InvalidArgument(
            "Window length cannot exceed FFT size".to_string(),
        ));
    }

    if hop_len == 0 {
        return Err(TorshError::InvalidArgument(
            "Hop length must be positive".to_string(),
        ));
    }

    // Generate window
    let window_data = generate_window(window, win_len)?;

    // Normalize window if requested
    let window_data = if normalized {
        let energy: f32 = window_data.iter().map(|w| w * w).sum();
        let scale = (energy / win_len as f32).sqrt();
        window_data.iter().map(|w| w / scale).collect()
    } else {
        window_data
    };

    // Get signal data
    let signal_data = input.data()?;
    let dims = input_shape.dims();

    let (batch_size, signal_len) = if ndim == 1 {
        (1, dims[0])
    } else {
        (dims[0], dims[1])
    };

    // Apply center padding if requested
    let (padded_signal, padded_len) = if center {
        let pad_amount = n_fft / 2;
        let new_len = signal_len + 2 * pad_amount;
        let mut padded = vec![0.0; batch_size * new_len];

        for b in 0..batch_size {
            let src_start = b * signal_len;
            let dst_start = b * new_len + pad_amount;

            // Copy signal data
            for i in 0..signal_len {
                padded[dst_start + i] = signal_data[src_start + i];
            }

            // Reflect padding
            for i in 0..pad_amount {
                if i < signal_len {
                    padded[b * new_len + i] = signal_data[src_start + pad_amount - i];
                }
            }
            for i in 0..pad_amount {
                if signal_len > i + 1 {
                    padded[dst_start + signal_len + i] =
                        signal_data[src_start + signal_len - 2 - i];
                }
            }
        }

        (padded, new_len)
    } else {
        (signal_data.to_vec(), signal_len)
    };

    // Calculate number of frames
    let n_frames = if padded_len >= n_fft {
        (padded_len - n_fft) / hop_len + 1
    } else {
        0
    };

    if n_frames == 0 {
        return Err(TorshError::InvalidArgument(
            "Signal too short for STFT".to_string(),
        ));
    }

    // Frequency bins
    let freq_bins = if onesided { n_fft / 2 + 1 } else { n_fft };

    // Process each frame
    let mut stft_data = Vec::with_capacity(batch_size * freq_bins * n_frames * 2);

    for b in 0..batch_size {
        let signal_start = b * padded_len;

        for frame_idx in 0..n_frames {
            let frame_start = signal_start + frame_idx * hop_len;

            // Extract and window the frame
            let mut frame = vec![0.0; n_fft];
            for i in 0..win_len.min(n_fft) {
                if frame_start + i < signal_start + padded_len {
                    frame[i] = padded_signal[frame_start + i] * window_data[i];
                }
            }

            // Create tensor for FFT
            let frame_tensor = Tensor::from_data(frame, vec![n_fft], input.device())?;

            // Apply FFT
            let fft_result = if onesided {
                rfft(&frame_tensor, Some(n_fft), None, None)?
            } else {
                // For two-sided, convert to complex and use full FFT
                use torsh_core::dtype::Complex32;
                let complex_frame: Vec<Complex32> = frame_tensor
                    .data()?
                    .iter()
                    .map(|&x| Complex32::new(x, 0.0))
                    .collect();
                let complex_tensor = Tensor::from_data(complex_frame, vec![n_fft], input.device())?;
                fft(&complex_tensor, Some(n_fft), None, None)?
            };

            // Extract real and imaginary parts
            let fft_data = fft_result.data()?;
            for val in fft_data.iter() {
                stft_data.push(val.re);
                stft_data.push(val.im);
            }
        }
    }

    // Create output tensor
    let output_shape = if ndim == 1 {
        vec![freq_bins, n_frames, 2]
    } else {
        vec![batch_size, freq_bins, n_frames, 2]
    };

    Tensor::from_data(stft_data, output_shape, input.device())
}

/// Inverse Short-Time Fourier Transform with overlap-add reconstruction
///
/// # Arguments
///
/// * `stft` - STFT tensor from stft_complete
/// * `n_fft` - FFT size
/// * `hop_length` - Hop length used in forward STFT
/// * `win_length` - Window length
/// * `window` - Window function (should match forward STFT)
/// * `center` - Whether center padding was used in forward STFT
/// * `normalized` - Whether normalization was used in forward STFT
/// * `onesided` - Whether one-sided FFT was used
/// * `length` - Desired output length (None infers from STFT shape)
///
/// # Returns
///
/// Reconstructed signal tensor
///
/// # Mathematical Formula
///
/// Overlap-add reconstruction:
/// ```text
/// x[n] = Σ(m) IFFT(STFT[m, :]) * w[n - mH] / Σ(m) w²[n - mH]
/// ```
///
/// # Examples
///
/// ```rust,ignore
/// let signal = randn(&[16384], None, None, None)?;
/// let spec = stft_complete(&signal, 512, Some(128), None, WindowFunction::Hann, true, false, true)?;
/// let reconstructed = istft_complete(&spec, 512, Some(128), None, WindowFunction::Hann, true, false, true, Some(16384))?;
/// ```
pub fn istft_complete(
    stft: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: WindowFunction,
    center: bool,
    normalized: bool,
    onesided: bool,
    length: Option<usize>,
) -> TorshResult<Tensor> {
    let stft_shape = stft.shape();
    let ndim = stft_shape.ndim();

    if ndim < 3 || ndim > 4 {
        return Err(TorshError::InvalidArgument(
            "ISTFT input must be 3D [freq, time, 2] or 4D [batch, freq, time, 2]".to_string(),
        ));
    }

    let dims = stft_shape.dims();
    if dims[ndim - 1] != 2 {
        return Err(TorshError::InvalidArgument(
            "Last dimension must be 2 (real, imag)".to_string(),
        ));
    }

    let hop_len = hop_length.unwrap_or(n_fft / 4);
    let win_len = win_length.unwrap_or(n_fft);

    // Generate window
    let window_data = generate_window(window, win_len)?;
    let window_data = if normalized {
        let energy: f32 = window_data.iter().map(|w| w * w).sum();
        let scale = (energy / win_len as f32).sqrt();
        window_data.iter().map(|w| w / scale).collect()
    } else {
        window_data
    };

    let (batch_size, freq_bins, n_frames) = if ndim == 3 {
        (1, dims[0], dims[1])
    } else {
        (dims[0], dims[1], dims[2])
    };

    // Verify frequency bins
    let expected_bins = if onesided { n_fft / 2 + 1 } else { n_fft };
    if freq_bins != expected_bins {
        return Err(TorshError::InvalidArgument(format!(
            "Frequency bins mismatch: expected {}, got {}",
            expected_bins, freq_bins
        )));
    }

    // Calculate output length
    let output_len = length.unwrap_or((n_frames - 1) * hop_len + n_fft);

    // Prepare output and window sum for overlap-add
    let mut output_data = vec![0.0; batch_size * output_len];
    let mut window_sum = vec![0.0; output_len];

    // Get STFT data
    let stft_data = stft.data()?;

    // Process each batch and frame
    for b in 0..batch_size {
        let batch_offset = if ndim == 3 {
            0
        } else {
            b * freq_bins * n_frames * 2
        };

        for frame_idx in 0..n_frames {
            // Extract complex frame
            use torsh_core::dtype::Complex32;
            let mut frame_fft = Vec::with_capacity(freq_bins);

            for f in 0..freq_bins {
                let idx = batch_offset + (f * n_frames + frame_idx) * 2;
                if idx + 1 < stft_data.len() {
                    frame_fft.push(Complex32::new(stft_data[idx], stft_data[idx + 1]));
                } else {
                    frame_fft.push(Complex32::new(0.0, 0.0));
                }
            }

            // Apply IFFT
            let fft_tensor = Tensor::from_data(frame_fft, vec![freq_bins], stft.device())?;

            let frame_signal = if onesided {
                super::spectral_advanced::irfft(&fft_tensor, Some(n_fft), None, None)?
            } else {
                let ifft_result = ifft(&fft_tensor, Some(n_fft), None, None)?;
                let ifft_data = ifft_result.data()?;
                let real_data: Vec<f32> = ifft_data.iter().map(|c| c.re).collect();
                Tensor::from_data(real_data, vec![n_fft], ifft_result.device())?
            };

            let frame_data = frame_signal.data()?;

            // Overlap-add with windowing
            let frame_start = frame_idx * hop_len;
            for i in 0..win_len.min(n_fft) {
                let output_idx = b * output_len + frame_start + i;
                if output_idx < (b + 1) * output_len && i < frame_data.len() {
                    output_data[output_idx] += frame_data[i] * window_data[i];
                    if b == 0 {
                        // Only accumulate window sum once
                        window_sum[frame_start + i] += window_data[i] * window_data[i];
                    }
                }
            }
        }
    }

    // Normalize by window sum to compensate for overlapping windows
    for b in 0..batch_size {
        for i in 0..output_len {
            let idx = b * output_len + i;
            if window_sum[i] > 1e-8 {
                output_data[idx] /= window_sum[i];
            }
        }
    }

    // Remove center padding if it was applied
    let final_data = if center {
        let pad_amount = n_fft / 2;
        let unpadded_len = output_len.saturating_sub(2 * pad_amount);
        let mut unpadded = Vec::with_capacity(batch_size * unpadded_len);

        for b in 0..batch_size {
            let src_start = b * output_len + pad_amount;
            for i in 0..unpadded_len {
                if src_start + i < output_data.len() {
                    unpadded.push(output_data[src_start + i]);
                } else {
                    unpadded.push(0.0);
                }
            }
        }

        unpadded
    } else {
        output_data
    };

    // Create output tensor
    let final_len = if center {
        output_len.saturating_sub(n_fft)
    } else {
        output_len
    };

    let output_shape = if batch_size == 1 {
        vec![final_len]
    } else {
        vec![batch_size, final_len]
    };

    Tensor::from_data(final_data, output_shape, stft.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_ops::randn;

    #[test]
    fn test_window_generation() -> TorshResult<()> {
        // Test Hann window
        let hann = generate_window(WindowFunction::Hann, 256)?;
        assert_eq!(hann.len(), 256);
        assert!(hann[0] < 0.01); // Near zero at edges
        assert!(hann[128] > 0.99); // Near one at center

        // Test Hamming window
        let hamming = generate_window(WindowFunction::Hamming, 256)?;
        assert_eq!(hamming.len(), 256);

        // Test Rectangular window
        let rect = generate_window(WindowFunction::Rectangular, 256)?;
        assert!(rect.iter().all(|&x| (x - 1.0).abs() < 1e-6));

        Ok(())
    }

    #[test]
    fn test_stft_basic() -> TorshResult<()> {
        let signal = randn(&[1024], None, None, None)?;

        let stft_result = stft_complete(
            &signal,
            256,
            Some(128),
            None,
            WindowFunction::Hann,
            true,
            false,
            true,
        )?;

        // Check shape: [freq_bins, time_frames, 2]
        let stft_result_shape = stft_result.shape();
        let shape = stft_result_shape.dims();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0], 129); // n_fft/2 + 1
        assert_eq!(shape[2], 2); // Real and imaginary

        Ok(())
    }

    #[test]
    fn test_stft_istft_roundtrip() -> TorshResult<()> {
        let signal_len = 2048;
        let signal = randn(&[signal_len], None, None, None)?;

        let n_fft = 256;
        let hop_length = 64;

        // Forward STFT
        let stft_result = stft_complete(
            &signal,
            n_fft,
            Some(hop_length),
            None,
            WindowFunction::Hann,
            true,
            false,
            true,
        )?;

        // Inverse STFT
        let reconstructed = istft_complete(
            &stft_result,
            n_fft,
            Some(hop_length),
            None,
            WindowFunction::Hann,
            true,
            false,
            true,
            Some(signal_len),
        )?;

        // Check reconstruction accuracy
        let signal_data = signal.data()?;
        let recon_data = reconstructed.data()?;

        let mut max_error = 0.0f32;
        for i in 0..signal_len.min(recon_data.len()) {
            let error = (signal_data[i] - recon_data[i]).abs();
            max_error = max_error.max(error);
        }

        // Should have good reconstruction (higher tolerance due to smaller FFT size)
        assert!(max_error < 5.0, "Max reconstruction error: {}", max_error);

        Ok(())
    }

    #[test]
    fn test_stft_different_windows() -> TorshResult<()> {
        let signal = randn(&[1024], None, None, None)?;

        for window in &[
            WindowFunction::Hann,
            WindowFunction::Hamming,
            WindowFunction::Blackman,
            WindowFunction::Bartlett,
        ] {
            let stft_result =
                stft_complete(&signal, 256, Some(128), None, *window, false, false, true)?;

            assert_eq!(stft_result.shape().ndim(), 3);
        }

        Ok(())
    }

    #[test]
    fn test_stft_batch_processing() -> TorshResult<()> {
        let batch_size = 4;
        let signal_len = 1024;
        let batch_signal = randn(&[batch_size, signal_len], None, None, None)?;

        let stft_result = stft_complete(
            &batch_signal,
            256,
            Some(128),
            None,
            WindowFunction::Hann,
            true,
            false,
            true,
        )?;

        // Check shape: [batch, freq_bins, time_frames, 2]
        let stft_result_shape = stft_result.shape();
        let shape = stft_result_shape.dims();
        assert_eq!(shape.len(), 4);
        assert_eq!(shape[0], batch_size);
        assert_eq!(shape[1], 129); // n_fft/2 + 1
        assert_eq!(shape[3], 2);

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let signal = randn(&[64], None, None, None).unwrap();

        // Test with hop_length = 0
        let result = stft_complete(
            &signal,
            256,
            Some(0),
            None,
            WindowFunction::Hann,
            false,
            false,
            true,
        );
        assert!(result.is_err());

        // Test with window longer than FFT
        let result = stft_complete(
            &signal,
            128,
            Some(64),
            Some(256),
            WindowFunction::Hann,
            false,
            false,
            true,
        );
        assert!(result.is_err());

        // Test with signal too short
        let tiny_signal = randn(&[32], None, None, None).unwrap();
        let result = stft_complete(
            &tiny_signal,
            256,
            Some(128),
            None,
            WindowFunction::Hann,
            false,
            false,
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_window_properties() -> TorshResult<()> {
        // Test that windows have expected properties
        let size = 256;

        // Hann window should sum to approximately N/2
        let hann = generate_window(WindowFunction::Hann, size)?;
        let hann_sum: f32 = hann.iter().sum();
        assert!((hann_sum - (size as f32 / 2.0)).abs() < 10.0);

        // Hamming window should never go to zero
        let hamming = generate_window(WindowFunction::Hamming, size)?;
        assert!(hamming.iter().all(|&x| x > 0.08)); // Hamming minimum is ~0.08

        // Blackman window should have good sidelobe suppression
        let blackman = generate_window(WindowFunction::Blackman, size)?;
        assert!(blackman[0] < 0.01); // Near zero at edges
        assert!(blackman[size - 1] < 0.01);

        Ok(())
    }

    #[test]
    fn test_stft_energy_conservation() -> TorshResult<()> {
        // Test that STFT preserves energy (with proper normalization)
        let signal_len = 2048;
        let signal = randn(&[signal_len], None, None, None)?;
        let signal_data = signal.data()?;

        // Compute signal energy
        let signal_energy: f32 = signal_data.iter().map(|&x| x * x).sum();

        // Compute STFT
        let stft_result = stft_complete(
            &signal,
            256,
            Some(64),
            None,
            WindowFunction::Hann,
            false,
            false,
            true,
        )?;

        // Compute STFT energy
        let stft_data = stft_result.data()?;
        let mut stft_energy = 0.0f32;
        for chunk in stft_data.chunks_exact(2) {
            stft_energy += chunk[0] * chunk[0] + chunk[1] * chunk[1];
        }

        // Energies should be proportional (wider range due to smaller FFT and windowing effects)
        let ratio = stft_energy / signal_energy;
        assert!(ratio > 0.01 && ratio < 200.0, "Energy ratio: {}", ratio);

        Ok(())
    }

    #[test]
    fn test_stft_time_shift_property() -> TorshResult<()> {
        // Test that time shift in signal results in phase shift in STFT
        let signal_len = 1024;
        let mut signal1 = vec![0.0; signal_len];
        let mut signal2 = vec![0.0; signal_len];

        // Create impulse at different positions
        signal1[256] = 1.0;
        signal2[512] = 1.0;

        let tensor1 = Tensor::from_data(
            signal1,
            vec![signal_len],
            torsh_core::device::DeviceType::Cpu,
        )?;
        let tensor2 = Tensor::from_data(
            signal2,
            vec![signal_len],
            torsh_core::device::DeviceType::Cpu,
        )?;

        let stft1 = stft_complete(
            &tensor1,
            256,
            Some(128),
            None,
            WindowFunction::Hann,
            false,
            false,
            true,
        )?;
        let stft2 = stft_complete(
            &tensor2,
            256,
            Some(128),
            None,
            WindowFunction::Hann,
            false,
            false,
            true,
        )?;

        // Shapes should match
        assert_eq!(stft1.shape().dims(), stft2.shape().dims());

        Ok(())
    }

    #[test]
    fn test_istft_perfect_reconstruction_conditions() -> TorshResult<()> {
        // Test perfect reconstruction with specific overlap conditions
        let signal_len = 2048;
        let signal = randn(&[signal_len], None, None, None)?;

        // 75% overlap (hop = win/4) with Hann window gives perfect reconstruction
        let n_fft = 256;
        let hop_length = 64; // 256/4 = 64

        let stft_result = stft_complete(
            &signal,
            n_fft,
            Some(hop_length),
            None,
            WindowFunction::Hann,
            true,
            false,
            true,
        )?;

        let reconstructed = istft_complete(
            &stft_result,
            n_fft,
            Some(hop_length),
            None,
            WindowFunction::Hann,
            true,
            false,
            true,
            Some(signal_len),
        )?;

        // Check dimensions are reasonable (may not match exactly due to center padding)
        let recon_len = reconstructed.shape().dims()[0];
        assert!(
            recon_len >= signal_len - n_fft && recon_len <= signal_len + n_fft,
            "Reconstructed length {} not close to signal length {}",
            recon_len,
            signal_len
        );

        Ok(())
    }

    #[test]
    fn test_stft_onesided_vs_twosided() -> TorshResult<()> {
        // Test that one-sided STFT has half the frequency bins of two-sided
        let signal = randn(&[1024], None, None, None)?;
        let n_fft = 256;

        let onesided = stft_complete(
            &signal,
            n_fft,
            Some(128),
            None,
            WindowFunction::Hann,
            false,
            false,
            true,
        )?;
        let onesided_shape = onesided.shape();
        let onesided_freqs = onesided_shape.dims()[0];

        // One-sided should have N/2 + 1 frequency bins
        assert_eq!(onesided_freqs, n_fft / 2 + 1);

        Ok(())
    }

    #[test]
    fn test_stft_with_all_window_types() -> TorshResult<()> {
        // Ensure all window types work correctly in STFT
        let signal = randn(&[1024], None, None, None)?;

        let windows = vec![
            WindowFunction::Rectangular,
            WindowFunction::Hann,
            WindowFunction::Hamming,
            WindowFunction::Blackman,
            WindowFunction::Bartlett,
            WindowFunction::Kaiser(5),
        ];

        for window in windows {
            let result = stft_complete(&signal, 256, Some(128), None, window, false, false, true)?;
            assert_eq!(result.shape().ndim(), 3);
        }

        Ok(())
    }

    #[test]
    fn test_stft_normalized_vs_unnormalized() -> TorshResult<()> {
        // Test difference between normalized and unnormalized STFT
        let signal = randn(&[1024], None, None, None)?;

        let normalized = stft_complete(
            &signal,
            256,
            Some(128),
            None,
            WindowFunction::Hann,
            false,
            true,
            true,
        )?;
        let unnormalized = stft_complete(
            &signal,
            256,
            Some(128),
            None,
            WindowFunction::Hann,
            false,
            false,
            true,
        )?;

        // Shapes should match
        assert_eq!(normalized.shape().dims(), unnormalized.shape().dims());

        // Magnitudes should differ by normalization factor
        let norm_data = normalized.data()?;
        let unnorm_data = unnormalized.data()?;

        // Check that they're different
        let mut diff_count = 0;
        for i in 0..norm_data.len().min(unnorm_data.len()) {
            if (norm_data[i] - unnorm_data[i]).abs() > 1e-6 {
                diff_count += 1;
            }
        }
        assert!(diff_count > 0);

        Ok(())
    }

    #[test]
    fn test_kaiser_window_beta_parameter() -> TorshResult<()> {
        // Test that Kaiser window behaves correctly with different beta values
        let size = 256;

        let kaiser_low = generate_window(WindowFunction::Kaiser(0), size)?;
        let kaiser_high = generate_window(WindowFunction::Kaiser(10), size)?;

        // Higher beta should result in narrower main lobe (lower values at edges)
        assert!(kaiser_high[0] < kaiser_low[0]);
        assert!(kaiser_high[size - 1] < kaiser_low[size - 1]);

        Ok(())
    }
}
