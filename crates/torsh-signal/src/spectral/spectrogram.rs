//! Spectrogram computation utilities
//!
//! This module provides spectrogram computation functions with PyTorch compatibility.

use torsh_core::{dtype::Complex32, error::Result};
use torsh_tensor::Tensor;

use super::stft::{stft, StftParams};
use crate::windows::Window;

/// Compute a spectrogram using STFT
///
/// This function computes the magnitude spectrogram of a signal using the Short-Time
/// Fourier Transform (STFT). It's compatible with PyTorch's `torch.spectrogram`.
///
/// # Arguments
///
/// * `input` - Input signal tensor
/// * `n_fft` - FFT size
/// * `hop_length` - Hop length between frames
/// * `win_length` - Window length
/// * `window` - Window function type
/// * `center` - Whether to center the signal
/// * `pad_mode` - Padding mode (currently unused)
/// * `normalized` - Whether to normalize the FFT
/// * `onesided` - Whether to return onesided spectrum
/// * `power` - Power to raise the magnitude to (None, Some(1.0), Some(2.0))
///
/// # Returns
///
/// Magnitude spectrogram tensor
pub fn spectrogram(
    input: &Tensor<f32>,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Option<Window>,
    center: bool,
    _pad_mode: &str,
    normalized: bool,
    onesided: bool,
    power: Option<f32>,
) -> Result<Tensor<f32>> {
    let params = StftParams {
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        normalized,
        onesided,
        return_complex: true,
    };

    let stft_result = stft(input, params)?;

    // Compute magnitude
    let magnitude = complex_magnitude(&stft_result)?;

    // Apply power scaling if requested
    match power {
        Some(p) if p != 1.0 => magnitude.pow_scalar(p),
        _ => Ok(magnitude),
    }
}

/// Compute magnitude of complex tensor
pub(crate) fn complex_magnitude(tensor: &Tensor<Complex32>) -> Result<Tensor<f32>> {
    // Compute magnitude of complex numbers: sqrt(re^2 + im^2)
    tensor
        .abs()
        .map_err(|e| torsh_core::error::TorshError::ComputeError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_spectrogram_basic() -> Result<()> {
        let signal = ones(&[1024])?;
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

        let shape = spec.shape();
        assert_eq!(shape.ndim(), 2);
        assert_eq!(shape.dims()[0], 257); // n_fft/2 + 1 for onesided

        Ok(())
    }

    #[test]
    fn test_spectrogram_power_scaling() -> Result<()> {
        let signal = ones(&[512])?;

        // Test different power values
        let spec_power1 = spectrogram(
            &signal,
            256,
            Some(128),
            Some(256),
            Some(Window::Hann),
            true,
            "reflect",
            false,
            true,
            Some(1.0),
        )?;

        let spec_power2 = spectrogram(
            &signal,
            256,
            Some(128),
            Some(256),
            Some(Window::Hann),
            true,
            "reflect",
            false,
            true,
            Some(2.0),
        )?;

        // Shapes should be the same
        assert_eq!(spec_power1.shape().dims(), spec_power2.shape().dims());

        Ok(())
    }
}
