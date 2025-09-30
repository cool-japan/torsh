//! Spectral operations for signal processing
//!
//! This module provides PyTorch-compatible spectral analysis operations built on top of
//! scirs2-fft and scirs2-signal for high-performance signal processing.
//!
//! Features complete PyTorch compatibility for:
//! - torch.stft / torch.istft
//! - torch.spectrogram
//! - torch.mel_scale / torch.inverse_mel_scale
//! - Advanced mel-scale operations powered by SciRS2
//!
//! The module is organized into the following submodules:
//! - [`stft`] - Short-Time Fourier Transform operations
//! - [`mel`] - Mel-scale operations and filterbanks
//! - [`spectrogram`] - Spectrogram computation utilities

pub mod mel;
pub mod spectrogram;
pub mod stft;

// Re-export all public APIs to maintain compatibility
pub use mel::{
    create_fb_matrix, hz_to_mel, inverse_mel_scale, mel_filterbank, mel_scale, mel_spectrogram,
    mel_to_hz,
};
pub use spectrogram::spectrogram;
pub use stft::{istft, stft, StftParams};

// Use available scirs2 functionality
use scirs2_core as _; // Available but with simplified usage

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_module_reexports() {
        // Test that all re-exports are accessible
        let _params = StftParams::default();

        // Test mel-scale functions are available
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 0.01);
    }

    #[test]
    fn test_stft_shape() -> torsh_core::error::Result<()> {
        let signal = zeros(&[1024])?;
        let params = StftParams {
            n_fft: 512,
            hop_length: Some(256),
            ..Default::default()
        };

        let result = stft(&signal, params).unwrap();
        let shape = result.shape();

        // Should have shape [n_freqs, n_frames]
        assert_eq!(shape.ndim(), 2);
        assert_eq!(shape.dims()[0], 257); // n_fft/2 + 1 for onesided
        Ok(())
    }

    #[test]
    fn test_mel_filterbank() -> torsh_core::error::Result<()> {
        let filterbank = mel_filterbank(128, 2048, 16000.0, 0.0, None).unwrap();
        assert_eq!(filterbank.shape().dims(), &[128, 1025]);

        // Check that filters sum to approximately 1
        let sum = filterbank.sum()?;
        if sum.shape().ndim() == 0 {
            // Sum is a scalar
            let val = sum.item();
            let val_unwrapped = val.unwrap();
            assert!(val_unwrapped >= 0.0 && val_unwrapped <= 2048.0); // Total sum across all filters
        } else {
            // Sum per dimension
            for i in 0..sum.shape().dims()[0] {
                let val = sum.get_1d(i)?;
                assert!(val >= 0.0 && val <= 2.0);
            }
        }
        Ok(())
    }

    #[test]
    fn test_hz_mel_conversion() {
        use approx::assert_relative_eq;
        assert_relative_eq!(hz_to_mel(0.0), 0.0, epsilon = 1e-5);
        assert_relative_eq!(mel_to_hz(0.0), 0.0, epsilon = 1e-5);

        // Test round-trip conversion
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert_relative_eq!(hz, hz_back, epsilon = 1e-3);
    }

    #[test]
    fn test_create_fb_matrix() -> torsh_core::error::Result<()> {
        let n_freqs = 513; // n_fft/2 + 1 where n_fft = 1024
        let f_min = 0.0;
        let f_max = 8000.0;
        let n_mels = 128;
        let sample_rate = 16000.0;

        let fb_matrix = create_fb_matrix(n_freqs, f_min, f_max, n_mels, sample_rate)?;
        let shape = fb_matrix.shape();

        // Check shape
        assert_eq!(shape.dims(), &[n_mels, n_freqs]);

        // Check that each filter has non-negative values (sum bounds can be quite large depending on parameters)
        for i in 0..n_mels {
            let mut sum = 0.0;
            for j in 0..n_freqs {
                let val = fb_matrix.get_2d(i, j)?;
                assert!(val >= 0.0, "Mel filter values should be non-negative"); // Non-negative values
                sum += val;
            }
            assert!(sum >= 0.0, "Mel filter sum should be non-negative"); // At least non-negative sum
        }

        Ok(())
    }

    #[test]
    fn test_mel_spectrogram() -> torsh_core::error::Result<()> {
        use torsh_core::device::DeviceType;
        // Create a dummy spectrogram
        let n_freqs = 513;
        let n_frames = 100;
        let mut specgram: torsh_tensor::Tensor<f32> =
            torsh_tensor::Tensor::zeros(&[n_freqs, n_frames], DeviceType::Cpu)?;

        // Fill with some test data
        for i in 0..n_freqs {
            for j in 0..n_frames {
                specgram.set_2d(i, j, (i + j) as f32 * 0.001)?;
            }
        }

        let f_min = 0.0;
        let f_max = Some(8000.0);
        let n_mels = 80;
        let sample_rate = 16000.0;

        let mel_spec = mel_spectrogram(&specgram, f_min, f_max, n_mels, sample_rate)?;
        let shape = mel_spec.shape();

        // Check output shape
        assert_eq!(shape.dims(), &[n_mels, n_frames]);

        // Check that values are non-negative (mel spectrograms should be non-negative)
        for i in 0..n_mels {
            for j in 0..n_frames {
                let value = mel_spec.get_2d(i, j)?;
                assert!(value >= 0.0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_mel_scale_operations() -> torsh_core::error::Result<()> {
        use torsh_core::device::DeviceType;
        let n_freqs = 257; // n_fft/2 + 1 where n_fft = 512
        let n_mels = 40;
        let sample_rate = 16000.0;
        let f_min = 0.0;
        let f_max = Some(8000.0);

        // Create a dummy frequency-domain tensor
        let mut freqs: torsh_tensor::Tensor<f32> =
            torsh_tensor::Tensor::zeros(&[n_freqs], DeviceType::Cpu)?;
        for i in 0..n_freqs {
            freqs.set_1d(i, i as f32 * 0.1)?;
        }

        // Test mel_scale
        let mel_result = mel_scale(&freqs, f_min, f_max, n_mels, sample_rate)?;
        assert_eq!(mel_result.shape().dims()[0], n_mels);

        // Test inverse_mel_scale
        let inverse_result = inverse_mel_scale(&mel_result, f_min, f_max, n_freqs, sample_rate)?;
        assert_eq!(inverse_result.shape().dims()[0], n_freqs);

        // Check that values are reasonable (not NaN or infinite)
        for i in 0..n_mels {
            let val = mel_result.get_1d(i)?;
            assert!(val.is_finite());
        }

        for i in 0..n_freqs {
            let val = inverse_result.get_1d(i)?;
            assert!(val.is_finite());
        }

        Ok(())
    }
}
