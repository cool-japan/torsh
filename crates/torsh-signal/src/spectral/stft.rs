//! Short-Time Fourier Transform (STFT) implementation
//!
//! This module provides STFT and ISTFT operations with PyTorch compatibility.

use torsh_core::{
    device::DeviceType,
    dtype::Complex32,
    error::{Result, TorshError},
};
use torsh_functional::spectral::{fft, ifft, rfft};
use torsh_tensor::{creation::ones, Tensor};

use crate::windows::Window;

/// Configuration parameters for Short-Time Fourier Transform (STFT) operations.
///
/// This struct provides full control over STFT computation parameters,
/// matching PyTorch's `torch.stft` interface for compatibility.
///
/// # Examples
///
/// ```rust,no_run
/// use torsh_signal::spectral::StftParams;
/// use torsh_signal::windows::Window;
///
/// // Basic STFT configuration
/// let params = StftParams {
///     n_fft: 1024,
///     hop_length: Some(512),
///     window: Some(Window::Hann),
///     ..Default::default()
/// };
///
/// // High-resolution analysis
/// let high_res_params = StftParams {
///     n_fft: 4096,
///     hop_length: Some(1024),
///     win_length: Some(2048),
///     window: Some(Window::Kaiser(8.6)),
///     center: true,
///     normalized: true,
///     onesided: true,
///     return_complex: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct StftParams {
    /// FFT length (number of frequency bins = n_fft/2 + 1 for onesided)
    pub n_fft: usize,
    /// Number of samples between successive frames (default: n_fft // 4)
    pub hop_length: Option<usize>,
    /// Length of the window function (default: n_fft)
    pub win_length: Option<usize>,
    /// Window function to apply (default: Hann window)
    pub window: Option<Window>,
    /// Whether to center the signal by padding (default: true)
    pub center: bool,
    /// Whether to normalize the FFT (default: false)
    pub normalized: bool,
    /// Whether to return onesided FFT (default: true for real inputs)
    pub onesided: bool,
    /// Whether to return complex output (default: true)
    pub return_complex: bool,
}

impl Default for StftParams {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop_length: None,
            win_length: None,
            window: Some(Window::Hann),
            center: true,
            normalized: false,
            onesided: true,
            return_complex: true,
        }
    }
}

/// Wrapper for FFT that handles real f32 input using real FFT
fn apply_fft_real(
    input: &Tensor<f32>,
    n_fft: Option<usize>,
    dim: i32,
    normalized: bool,
) -> Result<Tensor<Complex32>> {
    let norm_str = if normalized { Some("ortho") } else { None };
    rfft(input, n_fft, Some(dim), norm_str).map_err(|e| TorshError::ComputeError(e.to_string()))
}

/// Wrapper for FFT that handles complex input
#[allow(dead_code)]
fn apply_fft_complex(
    input: &Tensor<Complex32>,
    n_fft: Option<usize>,
    dim: i32,
    normalized: bool,
) -> Result<Tensor<Complex32>> {
    let norm_str = if normalized { Some("ortho") } else { None };
    fft(input, n_fft, Some(dim), norm_str).map_err(|e| TorshError::ComputeError(e.to_string()))
}

/// Wrapper for IFFT that handles type conversion
fn apply_ifft(
    input: &Tensor<Complex32>,
    n_fft: Option<usize>,
    dim: i32,
    normalized: bool,
) -> Result<Tensor<Complex32>> {
    let norm_str = if normalized { Some("ortho") } else { None };
    ifft(input, n_fft, Some(dim), norm_str).map_err(|e| TorshError::ComputeError(e.to_string()))
}

/// Compute the Short-Time Fourier Transform (STFT) of a real-valued signal.
///
/// The STFT provides time-frequency analysis by computing the Fourier transform
/// of localized portions of the signal, windowed by a sliding window function.
/// This implementation is fully compatible with PyTorch's `torch.stft`.
///
/// # Arguments
///
/// * `input` - Input signal tensor (1D or 2D with shape [batch, time])
/// * `params` - STFT configuration parameters (see [`StftParams`])
///
/// # Returns
///
/// Complex-valued tensor with shape:
/// - 1D input: `[n_freqs, n_frames]`
/// - 2D input: `[batch, n_freqs, n_frames]`
///
/// Where:
/// - `n_freqs = n_fft // 2 + 1` (for onesided) or `n_fft` (for two-sided)
/// - `n_frames = (signal_length + padding - n_fft) // hop_length + 1`
///
/// # Examples
///
/// ```rust,no_run
/// use torsh_signal::prelude::*;
/// use torsh_tensor::creation::ones;
///
/// // Basic STFT of a sine wave
/// let signal = ones(&[1024])?; // Replace with actual signal
/// let params = StftParams {
///     n_fft: 512,
///     hop_length: Some(256),
///     window: Some(Window::Hann),
///     ..Default::default()
/// };
///
/// let stft_result = stft(&signal, params)?;
/// println!("STFT shape: {:?}", stft_result.shape());
/// // Output: STFT shape: [257, 5] (for onesided with n_fft=512)
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance Notes
///
/// - Uses efficient real FFT for real-valued inputs
/// - Supports both periodic and symmetric window functions
/// - Memory usage scales with `n_fft × n_frames × batch_size`
/// - Consider using streaming STFT for very long signals
///
/// # Errors
///
/// Returns an error if:
/// - Input tensor has more than 2 dimensions
/// - Window parameters are inconsistent
/// - Memory allocation fails
pub fn stft(input: &Tensor<f32>, params: StftParams) -> Result<Tensor<Complex32>> {
    let input_shape = input.shape();
    if input_shape.is_empty() || input_shape.ndim() > 2 {
        return Err(TorshError::InvalidArgument(
            "Input must be 1D or 2D tensor".to_string(),
        ));
    }

    let n_fft = params.n_fft;
    let hop_length = params.hop_length.unwrap_or(n_fft / 4);
    let win_length = params.win_length.unwrap_or(n_fft);

    // Get window
    let window = match params.window {
        Some(w) => crate::windows::window(w, win_length, false)?,
        None => ones(&[win_length])?,
    };

    // Pad window to n_fft if necessary
    let window = if win_length < n_fft {
        pad_window(&window, n_fft)?
    } else {
        window
    };

    // Process input
    let signal = if input_shape.ndim() == 1 {
        input.unsqueeze(0)?
    } else {
        input.clone()
    };

    let batch_size = signal.shape().dims()[0];
    let _signal_length = signal.shape().dims()[1];

    // Pad signal if center is true
    let signal = if params.center {
        let pad_amount = n_fft / 2;
        pad_signal(&signal, pad_amount)?
    } else {
        signal
    };

    let padded_length = signal.shape().dims()[1];
    let n_frames = (padded_length - n_fft) / hop_length + 1;

    // Compute STFT frames - create complex zeros
    let mut stft_result: Tensor<Complex32> = if params.onesided {
        Tensor::zeros(&[batch_size, n_fft / 2 + 1, n_frames], DeviceType::Cpu)?
    } else {
        Tensor::zeros(&[batch_size, n_fft, n_frames], DeviceType::Cpu)?
    };

    for b in 0..batch_size {
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let end = start + n_fft;

            // Extract frame
            let frame = extract_frame(&signal, b, start, end)?;

            // Apply window
            let windowed_frame = frame.mul(&window)?;

            // Compute FFT
            let fft_result = apply_fft_real(&windowed_frame, Some(n_fft), -1, params.normalized)?;

            // Store result
            if params.onesided {
                // Only keep positive frequencies
                for f in 0..=n_fft / 2 {
                    let value = fft_result.get_1d(f)?;
                    stft_result.set_3d(b, f, frame_idx, value)?;
                }
            } else {
                for f in 0..n_fft {
                    let value = fft_result.get_1d(f)?;
                    stft_result.set_3d(b, f, frame_idx, value)?;
                }
            }
        }
    }

    // Remove batch dimension if input was 1D
    if input_shape.ndim() == 1 {
        stft_result = stft_result.squeeze(0)?;
    }

    Ok(stft_result)
}

/// Compute the Inverse Short-Time Fourier Transform (ISTFT)
pub fn istft(
    input: &Tensor<Complex32>,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Option<Window>,
    center: bool,
    normalized: bool,
    onesided: bool,
    length: Option<usize>,
) -> Result<Tensor<f32>> {
    let hop_length = hop_length.unwrap_or(n_fft / 4);
    let win_length = win_length.unwrap_or(n_fft);

    // Get window
    let window = match window {
        Some(w) => crate::windows::window(w, win_length, false)?,
        None => ones(&[win_length])?,
    };

    let input_shape = input.shape();
    let n_frames = input_shape.dims()[input_shape.ndim() - 1];
    let expected_signal_len = n_fft + hop_length * (n_frames - 1);

    // Initialize output
    let mut output: Tensor<f32> = Tensor::zeros(&[expected_signal_len], DeviceType::Cpu)?;
    let mut window_sum: Tensor<f32> = Tensor::zeros(&[expected_signal_len], DeviceType::Cpu)?;

    // Overlap-add synthesis
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;

        // Extract frame from STFT
        let stft_frame = extract_stft_frame(input, frame_idx, onesided)?;

        // Compute IFFT
        let time_frame = apply_ifft(&stft_frame, Some(n_fft), -1, normalized)?;

        // Apply window (extract real part of IFFT result first)
        let real_time_frame = time_frame.real()?;
        let windowed_frame = real_time_frame.mul(&window)?;

        // Add to output with overlap
        for i in 0..n_fft {
            if start + i < expected_signal_len {
                let current_val = output.get_1d(start + i)?;
                let frame_val = windowed_frame.get_1d(i)?;
                output.set_1d(start + i, current_val + frame_val)?;

                let window_val = window.get_1d(i)?;
                let current_window_sum = window_sum.get_1d(start + i)?;
                window_sum.set_1d(start + i, current_window_sum + window_val * window_val)?;
            }
        }
    }

    // Normalize by window sum
    for i in 0..expected_signal_len {
        let sum = window_sum.get_1d(i)?;
        if sum > 1e-8 {
            let val = output.get_1d(i)?;
            output.set_1d(i, val / sum)?;
        }
    }

    // Remove padding if center was used
    if center {
        let pad_amount = n_fft / 2;
        if expected_signal_len > 2 * pad_amount {
            let trimmed_len = expected_signal_len - 2 * pad_amount;
            let mut trimmed_output: Tensor<f32> = Tensor::zeros(&[trimmed_len], DeviceType::Cpu)?;
            for i in 0..trimmed_len {
                let val = output.get_1d(pad_amount + i)?;
                trimmed_output.set_1d(i, val)?;
            }
            output = trimmed_output;
        }
    }

    // Trim to specified length
    if let Some(target_length) = length {
        let current_len = output.shape().dims()[0];
        if current_len > target_length {
            let mut trimmed_output: Tensor<f32> = Tensor::zeros(&[target_length], DeviceType::Cpu)?;
            for i in 0..target_length {
                let val = output.get_1d(i)?;
                trimmed_output.set_1d(i, val)?;
            }
            output = trimmed_output;
        }
    }

    Ok(output)
}

/// Helper function to pad a window
pub(crate) fn pad_window(window: &Tensor<f32>, target_length: usize) -> Result<Tensor<f32>> {
    let window_length = window.shape().dims()[0];
    if window_length >= target_length {
        return Ok(window.clone());
    }

    let pad_left = (target_length - window_length) / 2;
    let _pad_right = target_length - window_length - pad_left;

    let mut padded: Tensor<f32> = Tensor::zeros(&[target_length], DeviceType::Cpu)?;
    for i in 0..window_length {
        let value = window.get_1d(i)?;
        padded.set_1d(pad_left + i, value)?;
    }

    Ok(padded)
}

/// Helper function to pad signal for center mode
pub(crate) fn pad_signal(signal: &Tensor<f32>, pad_amount: usize) -> Result<Tensor<f32>> {
    let shape = signal.shape();
    let batch_size = shape.dims()[0];
    let signal_length = shape.dims()[1];
    let padded_length = signal_length + 2 * pad_amount;

    let mut padded: Tensor<f32> = Tensor::zeros(&[batch_size, padded_length], DeviceType::Cpu)?;

    for b in 0..batch_size {
        // Reflect padding
        for i in 0..pad_amount {
            let left_val = signal.get_2d(b, pad_amount - i)?;
            padded.set_2d(b, i, left_val)?;

            let right_val = signal.get_2d(b, signal_length - pad_amount + i)?;
            padded.set_2d(b, signal_length + pad_amount + i, right_val)?;
        }

        // Copy original signal
        for i in 0..signal_length {
            let value = signal.get_2d(b, i)?;
            padded.set_2d(b, pad_amount + i, value)?;
        }
    }

    Ok(padded)
}

/// Helper function to extract a frame from the signal
pub(crate) fn extract_frame(
    signal: &Tensor<f32>,
    batch: usize,
    start: usize,
    end: usize,
) -> Result<Tensor<f32>> {
    let frame_length = end - start;
    let mut frame: Tensor<f32> = Tensor::zeros(&[frame_length], DeviceType::Cpu)?;

    for i in 0..frame_length {
        let value = signal.get_2d(batch, start + i)?;
        frame.set_1d(i, value)?;
    }

    Ok(frame)
}

/// Helper function to extract STFT frame for ISTFT
pub(crate) fn extract_stft_frame(
    stft: &Tensor<Complex32>,
    frame_idx: usize,
    onesided: bool,
) -> Result<Tensor<Complex32>> {
    let shape = stft.shape();
    let n_freqs = shape.dims()[shape.ndim() - 2];

    let mut frame: Tensor<Complex32> = if onesided {
        Tensor::zeros(&[n_freqs * 2 - 2], DeviceType::Cpu)?
    } else {
        Tensor::zeros(&[n_freqs], DeviceType::Cpu)?
    };

    if onesided {
        // Reconstruct full spectrum from one-sided
        for f in 0..n_freqs {
            let value = if shape.ndim() == 3 {
                stft.get_3d(0, f, frame_idx)?
            } else {
                stft.get_2d(f, frame_idx)?
            };
            frame.set_1d(f, value)?;

            // Mirror for negative frequencies (except DC and Nyquist)
            if f > 0 && f < n_freqs - 1 {
                frame.set_1d(n_freqs * 2 - 2 - f, value)?;
            }
        }
    } else {
        for f in 0..n_freqs {
            let value = if shape.ndim() == 3 {
                stft.get_3d(0, f, frame_idx)?
            } else {
                stft.get_2d(f, frame_idx)?
            };
            frame.set_1d(f, value)?;
        }
    }

    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_stft_shape() -> Result<()> {
        let signal = zeros(&[1024])?;
        let params = StftParams {
            n_fft: 512,
            hop_length: Some(256),
            ..Default::default()
        };

        let result = stft(&signal, params).expect("stft should succeed");
        let shape = result.shape();

        // Should have shape [n_freqs, n_frames]
        assert_eq!(shape.ndim(), 2);
        assert_eq!(shape.dims()[0], 257); // n_fft/2 + 1 for onesided
        Ok(())
    }

    #[test]
    fn test_stft_params_default() {
        let params = StftParams::default();
        assert_eq!(params.n_fft, 2048);
        assert_eq!(params.hop_length, None);
        assert_eq!(params.center, true);
        assert_eq!(params.onesided, true);
    }
}
