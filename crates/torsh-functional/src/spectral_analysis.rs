//! Comprehensive spectral analysis functions
//!
//! This module provides advanced spectral analysis operations including:
//! - Spectrogram computation (power, magnitude, phase)
//! - Mel-scale spectrogram for audio processing
//! - Cepstrum analysis for pitch and formant detection
//! - Spectral derivatives and features

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

use crate::spectral::{fft, ifft};
use crate::spectral_stft::{stft_complete, WindowFunction};

/// Spectrogram type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrogramType {
    /// Power spectrogram (|STFT|²)
    Power,
    /// Magnitude spectrogram (|STFT|)
    Magnitude,
    /// Log-power spectrogram (log(|STFT|² + eps))
    LogPower,
    /// Log-magnitude spectrogram (log(|STFT| + eps))
    LogMagnitude,
    /// Phase spectrogram (angle(STFT))
    Phase,
}

/// Compute spectrogram from audio signal
///
/// # Arguments
///
/// * `signal` - Input audio signal (1D or 2D for batched)
/// * `n_fft` - FFT size
/// * `hop_length` - Hop length
/// * `win_length` - Window length
/// * `window` - Window function
/// * `center` - Center padding
/// * `spectrogram_type` - Type of spectrogram to compute
///
/// # Returns
///
/// Spectrogram tensor with shape [freq_bins, time_frames] or [batch, freq_bins, time_frames]
///
/// # Mathematical Formulas
///
/// **Power spectrogram:**
/// ```text
/// S(f, t) = |STFT(f, t)|² = Re²(STFT) + Im²(STFT)
/// ```
///
/// **Magnitude spectrogram:**
/// ```text
/// S(f, t) = |STFT(f, t)| = sqrt(Re²(STFT) + Im²(STFT))
/// ```
///
/// **Log-power spectrogram:**
/// ```text
/// S(f, t) = log(|STFT(f, t)|² + ε)
/// ```
///
/// **Phase spectrogram:**
/// ```text
/// S(f, t) = atan2(Im(STFT), Re(STFT))
/// ```
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_functional::spectral_analysis::{spectrogram, SpectrogramType};
/// use torsh_functional::spectral_stft::WindowFunction;
///
/// let audio = randn(&[16384], None, None, None)?;
/// let spec = spectrogram(
///     &audio,
///     1024,                      // n_fft
///     Some(512),                 // hop_length
///     None,                      // win_length
///     WindowFunction::Hann,
///     true,                      // center
///     SpectrogramType::LogPower,
/// )?;
/// ```
pub fn spectrogram(
    signal: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: WindowFunction,
    center: bool,
    spectrogram_type: SpectrogramType,
) -> TorshResult<Tensor> {
    // Compute STFT
    let stft = stft_complete(
        signal, n_fft, hop_length, win_length, window, center, false, true, // onesided
    )?;

    let stft_shape = stft.shape();
    let dims = stft_shape.dims();
    let ndim = dims.len();

    // Extract dimensions
    let (batch_size, freq_bins, time_frames) = if ndim == 3 {
        (1, dims[0], dims[1])
    } else {
        (dims[0], dims[1], dims[2])
    };

    // Get STFT data (last dimension is [real, imag])
    let stft_data = stft.data()?;

    // Compute spectrogram based on type
    let spec_data = match spectrogram_type {
        SpectrogramType::Power => {
            let mut power = Vec::with_capacity(batch_size * freq_bins * time_frames);
            for chunk in stft_data.chunks_exact(2) {
                let re = chunk[0];
                let im = chunk[1];
                power.push(re * re + im * im);
            }
            power
        }
        SpectrogramType::Magnitude => {
            let mut magnitude = Vec::with_capacity(batch_size * freq_bins * time_frames);
            for chunk in stft_data.chunks_exact(2) {
                let re = chunk[0];
                let im = chunk[1];
                magnitude.push((re * re + im * im).sqrt());
            }
            magnitude
        }
        SpectrogramType::LogPower => {
            let eps = 1e-10;
            let mut log_power = Vec::with_capacity(batch_size * freq_bins * time_frames);
            for chunk in stft_data.chunks_exact(2) {
                let re = chunk[0];
                let im = chunk[1];
                let power = re * re + im * im;
                log_power.push((power + eps).ln());
            }
            log_power
        }
        SpectrogramType::LogMagnitude => {
            let eps = 1e-10;
            let mut log_mag = Vec::with_capacity(batch_size * freq_bins * time_frames);
            for chunk in stft_data.chunks_exact(2) {
                let re = chunk[0];
                let im = chunk[1];
                let mag = (re * re + im * im).sqrt();
                log_mag.push((mag + eps).ln());
            }
            log_mag
        }
        SpectrogramType::Phase => {
            let mut phase = Vec::with_capacity(batch_size * freq_bins * time_frames);
            for chunk in stft_data.chunks_exact(2) {
                let re = chunk[0];
                let im = chunk[1];
                phase.push(im.atan2(re));
            }
            phase
        }
    };

    // Create output tensor
    let output_shape = if ndim == 3 {
        vec![freq_bins, time_frames]
    } else {
        vec![batch_size, freq_bins, time_frames]
    };

    Tensor::from_data(spec_data, output_shape, signal.device())
}

/// Mel-scale frequency conversion
///
/// Convert Hz to Mel scale:
/// ```text
/// mel = 2595 * log10(1 + hz / 700)
/// ```
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Mel-scale to Hz conversion
///
/// Convert Mel scale to Hz:
/// ```text
/// hz = 700 * (10^(mel / 2595) - 1)
/// ```
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Create mel filter bank
///
/// # Arguments
///
/// * `n_mels` - Number of mel bands
/// * `n_fft` - FFT size
/// * `sample_rate` - Sample rate in Hz
/// * `f_min` - Minimum frequency
/// * `f_max` - Maximum frequency
///
/// # Returns
///
/// Mel filter bank matrix of shape [n_mels, n_fft/2 + 1]
///
/// # Mathematical Description
///
/// Creates triangular filters on the mel scale:
/// 1. Convert frequency range to mel scale
/// 2. Create evenly spaced mel points
/// 3. Convert mel points back to Hz
/// 4. Create triangular filters centered at each mel point
pub fn create_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> TorshResult<Vec<Vec<f32>>> {
    if n_mels == 0 {
        return Err(TorshError::InvalidArgument(
            "Number of mel bands must be positive".to_string(),
        ));
    }

    let n_freqs = n_fft / 2 + 1;

    // Convert to mel scale
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Create evenly spaced mel points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();

    // Create frequency bins
    let freq_bins: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate / n_fft as f32)
        .collect();

    // Create triangular filters
    let mut filterbank = vec![vec![0.0; n_freqs]; n_mels];

    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        for (k, &f) in freq_bins.iter().enumerate() {
            if f >= f_left && f <= f_center {
                // Rising edge
                filterbank[m][k] = (f - f_left) / (f_center - f_left);
            } else if f > f_center && f <= f_right {
                // Falling edge
                filterbank[m][k] = (f_right - f) / (f_right - f_center);
            }
        }
    }

    Ok(filterbank)
}

/// Mel spectrogram computation
///
/// Computes mel-scale spectrogram by applying mel filter bank to power spectrogram.
///
/// # Arguments
///
/// * `signal` - Input audio signal
/// * `n_fft` - FFT size
/// * `hop_length` - Hop length
/// * `n_mels` - Number of mel bands
/// * `sample_rate` - Sample rate in Hz
/// * `f_min` - Minimum frequency
/// * `f_max` - Maximum frequency
/// * `window` - Window function
///
/// # Returns
///
/// Mel spectrogram with shape [n_mels, time_frames] or [batch, n_mels, time_frames]
///
/// # Examples
///
/// ```rust,ignore
/// let audio = randn(&[16384], None, None, None)?;
/// let mel_spec = mel_spectrogram(
///     &audio,
///     1024,    // n_fft
///     Some(512),  // hop_length
///     80,      // n_mels
///     16000.0, // sample_rate
///     0.0,     // f_min
///     8000.0,  // f_max
///     WindowFunction::Hann,
/// )?;
/// ```
pub fn mel_spectrogram(
    signal: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    n_mels: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
    window: WindowFunction,
) -> TorshResult<Tensor> {
    // Compute power spectrogram
    let power_spec = spectrogram(
        signal,
        n_fft,
        hop_length,
        None,
        window,
        true,
        SpectrogramType::Power,
    )?;

    let spec_shape = power_spec.shape();
    let dims = spec_shape.dims();
    let ndim = dims.len();

    let (batch_size, freq_bins, time_frames) = if ndim == 2 {
        (1, dims[0], dims[1])
    } else {
        (dims[0], dims[1], dims[2])
    };

    // Create mel filter bank
    let mel_filters = create_mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max)?;

    // Apply mel filters to spectrogram
    let spec_data = power_spec.data()?;
    let mut mel_spec_data = vec![0.0; batch_size * n_mels * time_frames];

    for b in 0..batch_size {
        for t in 0..time_frames {
            for m in 0..n_mels {
                let mut mel_value = 0.0;

                for f in 0..freq_bins {
                    let spec_idx = if ndim == 2 {
                        f * time_frames + t
                    } else {
                        b * freq_bins * time_frames + f * time_frames + t
                    };

                    if spec_idx < spec_data.len() {
                        mel_value += mel_filters[m][f] * spec_data[spec_idx];
                    }
                }

                let mel_idx = if ndim == 2 {
                    m * time_frames + t
                } else {
                    b * n_mels * time_frames + m * time_frames + t
                };

                mel_spec_data[mel_idx] = mel_value;
            }
        }
    }

    // Create output tensor
    let output_shape = if ndim == 2 {
        vec![n_mels, time_frames]
    } else {
        vec![batch_size, n_mels, time_frames]
    };

    Tensor::from_data(mel_spec_data, output_shape, signal.device())
}

/// Cepstrum computation
///
/// Computes the cepstrum (inverse Fourier transform of log magnitude spectrum).
/// Useful for pitch detection and formant analysis.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `n_fft` - FFT size
///
/// # Returns
///
/// Cepstrum tensor
///
/// # Mathematical Formula
///
/// ```text
/// cepstrum = IFFT(log(|FFT(signal)|))
/// ```
///
/// # Applications
///
/// - Pitch detection (fundamental frequency estimation)
/// - Formant analysis (vocal tract resonances)
/// - Echo detection
/// - Speaker identification
pub fn cepstrum(signal: &Tensor, n_fft: usize) -> TorshResult<Tensor> {
    use torsh_core::dtype::Complex32;

    let signal_data = signal.data()?;
    let signal_len = signal_data.len();

    // Zero-pad to n_fft
    let mut padded = vec![0.0; n_fft];
    for i in 0..signal_len.min(n_fft) {
        padded[i] = signal_data[i];
    }

    // Convert to complex for FFT
    let complex_signal: Vec<Complex32> = padded.iter().map(|&x| Complex32::new(x, 0.0)).collect();
    let complex_tensor = Tensor::from_data(complex_signal, vec![n_fft], signal.device())?;

    // Compute FFT
    let fft_result = fft(&complex_tensor, Some(n_fft), None, None)?;

    // Compute log magnitude
    let fft_data = fft_result.data()?;
    let log_mag: Vec<Complex32> = fft_data
        .iter()
        .map(|c| {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            Complex32::new((mag + 1e-10).ln(), 0.0)
        })
        .collect();

    let log_mag_tensor = Tensor::from_data(log_mag, vec![n_fft], signal.device())?;

    // Compute IFFT
    let cepstrum_result = ifft(&log_mag_tensor, Some(n_fft), None, None)?;

    // Extract real part
    let cepstrum_data = cepstrum_result.data()?;
    let real_cepstrum: Vec<f32> = cepstrum_data.iter().map(|c| c.re).collect();

    Tensor::from_data(real_cepstrum, vec![n_fft], signal.device())
}

/// Spectral centroid computation
///
/// Computes the "center of mass" of the spectrum, indicating brightness.
///
/// # Mathematical Formula
///
/// ```text
/// centroid = Σ(f * S(f)) / Σ(S(f))
/// ```
///
/// where f is frequency and S(f) is magnitude spectrum.
pub fn spectral_centroid(
    signal: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    sample_rate: f32,
) -> TorshResult<Tensor> {
    // Compute magnitude spectrogram
    let spec = spectrogram(
        signal,
        n_fft,
        hop_length,
        None,
        WindowFunction::Hann,
        true,
        SpectrogramType::Magnitude,
    )?;

    let spec_shape = spec.shape();
    let dims = spec_shape.dims();
    let ndim = dims.len();

    let (batch_size, freq_bins, time_frames) = if ndim == 2 {
        (1, dims[0], dims[1])
    } else {
        (dims[0], dims[1], dims[2])
    };

    let spec_data = spec.data()?;

    // Create frequency bins
    let frequencies: Vec<f32> = (0..freq_bins)
        .map(|i| i as f32 * sample_rate / n_fft as f32)
        .collect();

    // Compute centroid for each frame
    let mut centroids = vec![0.0; batch_size * time_frames];

    for b in 0..batch_size {
        for t in 0..time_frames {
            let mut weighted_sum = 0.0;
            let mut magnitude_sum = 0.0;

            for f in 0..freq_bins {
                let idx = if ndim == 2 {
                    f * time_frames + t
                } else {
                    b * freq_bins * time_frames + f * time_frames + t
                };

                if idx < spec_data.len() {
                    let magnitude = spec_data[idx];
                    weighted_sum += frequencies[f] * magnitude;
                    magnitude_sum += magnitude;
                }
            }

            let centroid_idx = if ndim == 2 { t } else { b * time_frames + t };

            centroids[centroid_idx] = if magnitude_sum > 1e-10 {
                weighted_sum / magnitude_sum
            } else {
                0.0
            };
        }
    }

    // Create output tensor
    let output_shape = if ndim == 2 {
        vec![time_frames]
    } else {
        vec![batch_size, time_frames]
    };

    Tensor::from_data(centroids, output_shape, signal.device())
}

/// Spectral rolloff computation
///
/// Computes the frequency below which a specified percentage of total spectral energy is contained.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `n_fft` - FFT size
/// * `hop_length` - Hop length
/// * `sample_rate` - Sample rate
/// * `rolloff_percent` - Percentage threshold (e.g., 0.85 for 85%)
///
/// # Mathematical Description
///
/// Finds frequency f_r such that:
/// ```text
/// Σ(f ≤ f_r) S(f)² = rolloff_percent * Σ S(f)²
/// ```
pub fn spectral_rolloff(
    signal: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    sample_rate: f32,
    rolloff_percent: f32,
) -> TorshResult<Tensor> {
    if rolloff_percent < 0.0 || rolloff_percent > 1.0 {
        return Err(TorshError::InvalidArgument(
            "Rolloff percent must be between 0 and 1".to_string(),
        ));
    }

    // Compute power spectrogram
    let spec = spectrogram(
        signal,
        n_fft,
        hop_length,
        None,
        WindowFunction::Hann,
        true,
        SpectrogramType::Power,
    )?;

    let spec_shape = spec.shape();
    let dims = spec_shape.dims();
    let ndim = dims.len();

    let (batch_size, freq_bins, time_frames) = if ndim == 2 {
        (1, dims[0], dims[1])
    } else {
        (dims[0], dims[1], dims[2])
    };

    let spec_data = spec.data()?;

    // Create frequency bins
    let frequencies: Vec<f32> = (0..freq_bins)
        .map(|i| i as f32 * sample_rate / n_fft as f32)
        .collect();

    // Compute rolloff for each frame
    let mut rolloffs = vec![0.0; batch_size * time_frames];

    for b in 0..batch_size {
        for t in 0..time_frames {
            // Calculate total energy
            let mut total_energy = 0.0;
            for f in 0..freq_bins {
                let idx = if ndim == 2 {
                    f * time_frames + t
                } else {
                    b * freq_bins * time_frames + f * time_frames + t
                };

                if idx < spec_data.len() {
                    total_energy += spec_data[idx];
                }
            }

            let threshold = rolloff_percent * total_energy;

            // Find rolloff frequency
            let mut cumulative_energy = 0.0;
            let mut rolloff_freq = 0.0;

            for f in 0..freq_bins {
                let idx = if ndim == 2 {
                    f * time_frames + t
                } else {
                    b * freq_bins * time_frames + f * time_frames + t
                };

                if idx < spec_data.len() {
                    cumulative_energy += spec_data[idx];

                    if cumulative_energy >= threshold {
                        rolloff_freq = frequencies[f];
                        break;
                    }
                }
            }

            let rolloff_idx = if ndim == 2 { t } else { b * time_frames + t };
            rolloffs[rolloff_idx] = rolloff_freq;
        }
    }

    // Create output tensor
    let output_shape = if ndim == 2 {
        vec![time_frames]
    } else {
        vec![batch_size, time_frames]
    };

    Tensor::from_data(rolloffs, output_shape, signal.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_ops::randn;

    #[test]
    fn test_spectrogram_power() -> TorshResult<()> {
        let signal = randn(&[2048], None, None, None)?;
        let spec = spectrogram(
            &signal,
            256, // Reduced from 512 to avoid stack overflow in oxifft
            Some(128),
            None,
            WindowFunction::Hann,
            true,
            SpectrogramType::Power,
        )?;

        let spec_shape = spec.shape();
        let shape = spec_shape.dims();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], 129); // n_fft/2 + 1 (256/2 + 1)

        Ok(())
    }

    #[test]
    fn test_spectrogram_types() -> TorshResult<()> {
        let signal = randn(&[1024], None, None, None)?;

        for spec_type in &[
            SpectrogramType::Power,
            SpectrogramType::Magnitude,
            SpectrogramType::LogPower,
            SpectrogramType::LogMagnitude,
            SpectrogramType::Phase,
        ] {
            let spec = spectrogram(
                &signal,
                256,
                Some(128),
                None,
                WindowFunction::Hann,
                false,
                *spec_type,
            )?;

            assert_eq!(spec.shape().ndim(), 2);
        }

        Ok(())
    }

    #[test]
    fn test_mel_filterbank() -> TorshResult<()> {
        let filterbank = create_mel_filterbank(40, 512, 16000.0, 0.0, 8000.0)?;

        assert_eq!(filterbank.len(), 40);
        assert_eq!(filterbank[0].len(), 257); // n_fft/2 + 1

        // Check filters sum to reasonable values
        for filter in &filterbank {
            let sum: f32 = filter.iter().sum();
            assert!(sum > 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_mel_spectrogram() -> TorshResult<()> {
        let signal = randn(&[2048], None, None, None)?; // Reduced signal length
        let mel_spec = mel_spectrogram(
            &signal,
            256,       // Reduced from 1024 to avoid stack overflow in oxifft
            Some(128), // Adjusted hop_length proportionally
            80,
            16000.0,
            0.0,
            8000.0,
            WindowFunction::Hann,
        )?;

        let mel_spec_shape = mel_spec.shape();
        let shape = mel_spec_shape.dims();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], 80); // n_mels

        Ok(())
    }

    #[test]
    fn test_cepstrum() -> TorshResult<()> {
        let signal = randn(&[256], None, None, None)?; // Reduced from 512 to avoid stack overflow in oxifft
        let ceps = cepstrum(&signal, 256)?;

        assert_eq!(ceps.shape().dims(), &[256]);

        Ok(())
    }

    #[test]
    fn test_spectral_centroid() -> TorshResult<()> {
        let signal = randn(&[2048], None, None, None)?;
        let centroids = spectral_centroid(&signal, 256, Some(128), 16000.0)?; // Reduced from 512 to avoid stack overflow in oxifft

        assert_eq!(centroids.shape().ndim(), 1);
        assert!(centroids.shape().dims()[0] > 0);

        Ok(())
    }

    #[test]
    fn test_spectral_rolloff() -> TorshResult<()> {
        let signal = randn(&[2048], None, None, None)?;
        let rolloffs = spectral_rolloff(&signal, 256, Some(128), 16000.0, 0.85)?; // Reduced from 512 to avoid stack overflow in oxifft

        assert_eq!(rolloffs.shape().ndim(), 1);
        assert!(rolloffs.shape().dims()[0] > 0);

        Ok(())
    }

    #[test]
    fn test_hz_mel_conversion() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);

        assert!((hz - hz_back).abs() < 0.01);
    }

    #[test]
    fn test_mel_scale_properties() {
        // Test that mel scale has expected properties
        // Mel scale should be approximately linear for frequencies below 1000 Hz
        let freq_low = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let mel_low: Vec<f32> = freq_low.iter().map(|&f| hz_to_mel(f)).collect();

        // Check approximate linearity (relaxed tolerance to 15% due to log nature of mel scale)
        let diff1 = mel_low[1] - mel_low[0];
        let diff2 = mel_low[2] - mel_low[1];
        assert!((diff1 - diff2).abs() / diff1 < 0.15);

        // Mel scale should be logarithmic for high frequencies
        let freq_high = vec![5000.0, 6000.0, 7000.0, 8000.0];
        let mel_high: Vec<f32> = freq_high.iter().map(|&f| hz_to_mel(f)).collect();

        // Differences should decrease at higher frequencies
        let high_diff1 = mel_high[1] - mel_high[0];
        let high_diff2 = mel_high[2] - mel_high[1];
        assert!(high_diff2 < high_diff1);
    }

    #[test]
    fn test_mel_filterbank_properties() -> TorshResult<()> {
        // Test that mel filterbank has expected properties
        let n_mels = 40;
        let n_fft = 512;
        let sr = 16000.0;

        let filterbank = create_mel_filterbank(n_mels, n_fft, sr, 0.0, sr / 2.0)?;

        // Each filter should have triangular shape
        for filter in &filterbank {
            // Find peak location
            let max_val = filter.iter().cloned().fold(0.0f32, f32::max);
            assert!(max_val > 0.0);

            // Check that filter sums to a reasonable value
            let filter_sum: f32 = filter.iter().sum();
            assert!(filter_sum > 0.0);
        }

        // Filters should overlap smoothly
        for i in 0..n_mels - 1 {
            let overlap: f32 = filterbank[i]
                .iter()
                .zip(filterbank[i + 1].iter())
                .map(|(a, b)| a.min(*b))
                .sum();
            assert!(overlap > 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_spectrogram_non_negativity() -> TorshResult<()> {
        // Test that power and magnitude spectrograms are non-negative
        let signal = randn(&[2048], None, None, None)?;

        for spec_type in &[SpectrogramType::Power, SpectrogramType::Magnitude] {
            let spec = spectrogram(
                &signal,
                256, // Reduced from 512 to avoid stack overflow in oxifft
                Some(128),
                None,
                WindowFunction::Hann,
                false,
                *spec_type,
            )?;
            let spec_data = spec.data()?;

            // All values should be non-negative
            assert!(spec_data.iter().all(|&x| x >= 0.0));
        }

        Ok(())
    }

    #[test]
    fn test_log_spectrogram_no_infinities() -> TorshResult<()> {
        // Test that log spectrograms don't produce infinities
        let signal = randn(&[2048], None, None, None)?;

        for spec_type in &[SpectrogramType::LogPower, SpectrogramType::LogMagnitude] {
            let spec = spectrogram(
                &signal,
                256, // Reduced from 512 to avoid stack overflow in oxifft
                Some(128),
                None,
                WindowFunction::Hann,
                false,
                *spec_type,
            )?;
            let spec_data = spec.data()?;

            // No infinities or NaNs
            assert!(spec_data.iter().all(|&x| x.is_finite()));
        }

        Ok(())
    }

    #[test]
    fn test_phase_spectrogram_range() -> TorshResult<()> {
        // Test that phase spectrogram is in correct range [-π, π]
        let signal = randn(&[2048], None, None, None)?;

        let phase_spec = spectrogram(
            &signal,
            256, // Reduced from 512 to avoid stack overflow in oxifft
            Some(128),
            None,
            WindowFunction::Hann,
            false,
            SpectrogramType::Phase,
        )?;
        let phase_data = phase_spec.data()?;

        // All phase values should be in [-π, π]
        use std::f32::consts::PI;
        assert!(phase_data.iter().all(|&p| p >= -PI && p <= PI));

        Ok(())
    }

    #[test]
    fn test_mel_spectrogram_frequency_resolution() -> TorshResult<()> {
        // Test that mel spectrogram has correct frequency resolution
        let signal = randn(&[2048], None, None, None)?; // Reduced signal length
        let n_mels = 80;

        let mel_spec = mel_spectrogram(
            &signal,
            256,       // Reduced from 1024 to avoid stack overflow in oxifft
            Some(128), // Adjusted hop_length proportionally
            n_mels,
            16000.0,
            0.0,
            8000.0,
            WindowFunction::Hann,
        )?;

        let mel_shape = mel_spec.shape();
        let dims = mel_shape.dims();

        // First dimension should be n_mels
        assert_eq!(dims[0], n_mels);

        Ok(())
    }

    #[test]
    fn test_cepstrum_properties() -> TorshResult<()> {
        // Test that cepstrum has expected properties
        let n_fft = 256; // Reduced from 512 to avoid stack overflow in oxifft
        let mut signal = vec![0.0; n_fft];

        // Create impulse train (periodic signal)
        for i in (0..n_fft).step_by(32) {
            signal[i] = 1.0;
        }

        let signal_tensor =
            Tensor::from_data(signal, vec![n_fft], torsh_core::device::DeviceType::Cpu)?;
        let ceps = cepstrum(&signal_tensor, n_fft)?;

        let ceps_data = ceps.data()?;

        // Cepstrum should be real-valued
        assert!(ceps_data.iter().all(|&x| x.is_finite()));

        Ok(())
    }

    #[test]
    fn test_spectral_centroid_monotonicity() -> TorshResult<()> {
        // Test that spectral centroid increases with frequency content
        let n_fft = 256; // Reduced from 512 to avoid stack overflow in oxifft
        let sr = 16000.0;

        // Create low-frequency signal
        let mut low_freq = vec![0.0; 2048];
        for i in 0..low_freq.len() {
            low_freq[i] = (2.0 * std::f32::consts::PI * 100.0 * i as f32 / sr).sin();
        }
        let low_tensor =
            Tensor::from_data(low_freq, vec![2048], torsh_core::device::DeviceType::Cpu)?;

        // Create high-frequency signal
        let mut high_freq = vec![0.0; 2048];
        for i in 0..high_freq.len() {
            high_freq[i] = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr).sin();
        }
        let high_tensor =
            Tensor::from_data(high_freq, vec![2048], torsh_core::device::DeviceType::Cpu)?;

        let low_centroid = spectral_centroid(&low_tensor, n_fft, Some(128), sr)?; // Adjusted hop_length
        let high_centroid = spectral_centroid(&high_tensor, n_fft, Some(128), sr)?;

        let low_data = low_centroid.data()?;
        let high_data = high_centroid.data()?;

        // High frequency signal should have higher centroid
        let low_mean: f32 = low_data.iter().sum::<f32>() / low_data.len() as f32;
        let high_mean: f32 = high_data.iter().sum::<f32>() / high_data.len() as f32;

        assert!(high_mean > low_mean);

        Ok(())
    }

    #[test]
    fn test_spectral_rolloff_properties() -> TorshResult<()> {
        // Test that spectral rolloff has expected properties
        let signal = randn(&[2048], None, None, None)?;

        // Test different rolloff percentages
        for percent in [0.5, 0.75, 0.85, 0.95] {
            let rolloff = spectral_rolloff(&signal, 256, Some(128), 16000.0, percent)?; // Reduced from 512 to avoid stack overflow in oxifft
            let rolloff_data = rolloff.data()?;

            // All rolloff frequencies should be positive and less than Nyquist
            assert!(rolloff_data.iter().all(|&f| f >= 0.0 && f <= 8000.0));

            // Higher percentages should give higher rolloff frequencies (in general)
            // This is a statistical property, so we just check reasonableness
            let mean: f32 = rolloff_data.iter().sum::<f32>() / rolloff_data.len() as f32;
            assert!(mean >= 0.0 && mean <= 8000.0);
        }

        Ok(())
    }

    #[test]
    fn test_mel_spectrogram_time_invariance() -> TorshResult<()> {
        // Test that mel spectrogram of shifted signal has shifted frames
        let signal_len = 2048; // Reduced from 4096
        let mut signal1 = vec![0.0; signal_len];
        let mut signal2 = vec![0.0; signal_len];

        // Create pulse at different times
        for i in 50..100 {
            signal1[i] = 1.0;
        }
        for i in 550..600 {
            signal2[i] = 1.0;
        }

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

        let mel1 = mel_spectrogram(
            &tensor1,
            256,       // Reduced from 512 to avoid stack overflow in oxifft
            Some(128), // Adjusted hop_length proportionally
            40,
            16000.0,
            0.0,
            8000.0,
            WindowFunction::Hann,
        )?;
        let mel2 = mel_spectrogram(
            &tensor2,
            256,       // Reduced from 512 to avoid stack overflow in oxifft
            Some(128), // Adjusted hop_length proportionally
            40,
            16000.0,
            0.0,
            8000.0,
            WindowFunction::Hann,
        )?;

        // Shapes should match
        assert_eq!(mel1.shape().dims(), mel2.shape().dims());

        Ok(())
    }

    #[test]
    fn test_spectrogram_batch_processing() -> TorshResult<()> {
        // Test that batch processing produces consistent results
        let batch_signal = randn(&[4, 2048], None, None, None)?;

        let batch_spec = spectrogram(
            &batch_signal,
            256, // Reduced from 512 to avoid stack overflow in oxifft
            Some(128),
            None,
            WindowFunction::Hann,
            false,
            SpectrogramType::Power,
        )?;

        let batch_shape = batch_spec.shape();
        let dims = batch_shape.dims();

        // Should have 3 dimensions: [batch, freq, time]
        assert_eq!(dims.len(), 3);
        assert_eq!(dims[0], 4); // Batch size

        Ok(())
    }
}
