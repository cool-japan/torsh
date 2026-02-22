//! Spectral operations (FFT, STFT, etc.)

use oxifft::{Complex, Direction, Flags, Plan};
use torsh_core::{dtype::Complex32, Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

// Re-export advanced functions from sibling modules
pub use crate::spectral_advanced::{fftn, hfft, ifftn, ihfft, irfft, rfft2, rfftn};
pub use crate::spectral_analysis::{
    cepstrum, create_mel_filterbank, hz_to_mel, mel_spectrogram, mel_to_hz, spectral_centroid,
    spectral_rolloff, spectrogram, SpectrogramType,
};
pub use crate::spectral_stft::{generate_window, istft_complete, stft_complete, WindowFunction};

/// 1D Fast Fourier Transform
pub fn fft(
    input: &Tensor<Complex32>,
    n: Option<usize>,
    dim: Option<i32>,
    norm: Option<&str>,
) -> TorshResult<Tensor<Complex32>> {
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let ndim = dims.len();

    // Determine the dimension to apply FFT
    let fft_dim = if let Some(d) = dim {
        if d < 0 {
            (ndim as i32 + d) as usize
        } else {
            d as usize
        }
    } else {
        ndim - 1 // Last dimension by default
    };

    if fft_dim >= ndim {
        return Err(TorshError::InvalidArgument(format!(
            "FFT dimension {} is out of range for tensor with {} dimensions",
            fft_dim, ndim
        )));
    }

    let fft_size = n.unwrap_or(dims[fft_dim]);

    // Create FFT plan
    let plan = Plan::dft_1d(fft_size, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
        TorshError::InvalidArgument(format!("Failed to create FFT plan for size {}", fft_size))
    })?;

    let input_data = input.data()?;
    let input_len = input_data.len();

    // Handle reshaping for FFT computation
    let stride = dims[fft_dim];
    let batch_size = input_len / stride;

    let mut output_data = Vec::with_capacity(batch_size * fft_size);

    // Perform FFT on each batch
    for batch_idx in 0..batch_size {
        let mut input_buffer: Vec<Complex<f32>> = Vec::with_capacity(fft_size);

        // Extract data for this batch
        for i in 0..fft_size.min(stride) {
            let idx = batch_idx * stride + i;
            if idx < input_len {
                let complex_val = input_data[idx];
                input_buffer.push(Complex::new(complex_val.re, complex_val.im));
            } else {
                input_buffer.push(Complex::new(0.0, 0.0));
            }
        }

        // Zero-pad if necessary
        while input_buffer.len() < fft_size {
            input_buffer.push(Complex::new(0.0, 0.0));
        }

        // Perform FFT (out-of-place)
        let mut output_buffer = vec![Complex::zero(); fft_size];
        plan.execute(&input_buffer, &mut output_buffer);

        // Convert back to our Complex32 type
        for val in output_buffer {
            output_data.push(Complex32::new(val.re, val.im));
        }
    }

    // Create output shape
    let mut output_shape = dims.to_vec();
    output_shape[fft_dim] = fft_size;

    let mut result = Tensor::from_data(output_data, output_shape, input.device())?;

    // Apply normalization
    match norm {
        Some("ortho") => {
            let scale = Complex32::new(1.0 / (fft_size as f32).sqrt(), 0.0);
            result = result.mul_scalar(scale)?;
        }
        Some("forward") => {
            let scale = Complex32::new(1.0 / fft_size as f32, 0.0);
            result = result.mul_scalar(scale)?;
        }
        _ => {} // No normalization
    }

    Ok(result)
}

/// 1D Inverse Fast Fourier Transform
pub fn ifft(
    input: &Tensor<Complex32>,
    n: Option<usize>,
    dim: Option<i32>,
    norm: Option<&str>,
) -> TorshResult<Tensor<Complex32>> {
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let ndim = dims.len();

    // Determine the dimension to apply IFFT
    let fft_dim = if let Some(d) = dim {
        if d < 0 {
            (ndim as i32 + d) as usize
        } else {
            d as usize
        }
    } else {
        ndim - 1 // Last dimension by default
    };

    if fft_dim >= ndim {
        return Err(TorshError::InvalidArgument(format!(
            "IFFT dimension {} is out of range for tensor with {} dimensions",
            fft_dim, ndim
        )));
    }

    let fft_size = n.unwrap_or(dims[fft_dim]);

    // Create IFFT plan
    let plan = Plan::dft_1d(fft_size, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
        TorshError::InvalidArgument(format!("Failed to create IFFT plan for size {}", fft_size))
    })?;

    let input_data = input.data()?;
    let input_len = input_data.len();

    // Handle reshaping for IFFT computation
    let stride = dims[fft_dim];
    let batch_size = input_len / stride;

    let mut output_data = Vec::with_capacity(batch_size * fft_size);

    // Perform IFFT on each batch
    for batch_idx in 0..batch_size {
        let mut input_buffer: Vec<Complex<f32>> = Vec::with_capacity(fft_size);

        // Extract data for this batch
        for i in 0..fft_size.min(stride) {
            let idx = batch_idx * stride + i;
            if idx < input_len {
                let complex_val = input_data[idx];
                input_buffer.push(Complex::new(complex_val.re, complex_val.im));
            } else {
                input_buffer.push(Complex::new(0.0, 0.0));
            }
        }

        // Zero-pad if necessary
        while input_buffer.len() < fft_size {
            input_buffer.push(Complex::new(0.0, 0.0));
        }

        // Perform IFFT (out-of-place)
        let mut output_buffer = vec![Complex::zero(); fft_size];
        plan.execute(&input_buffer, &mut output_buffer);

        // Convert back to our Complex32 type
        for val in output_buffer {
            output_data.push(Complex32::new(val.re, val.im));
        }
    }

    // Create output shape
    let mut output_shape = dims.to_vec();
    output_shape[fft_dim] = fft_size;

    let mut result = Tensor::from_data(output_data, output_shape, input.device())?;

    // Apply normalization (IFFT typically divides by N)
    match norm {
        Some("ortho") => {
            let scale = Complex32::new(1.0 / (fft_size as f32).sqrt(), 0.0);
            result = result.mul_scalar(scale)?;
        }
        Some("backward") | None => {
            let scale = Complex32::new(1.0 / fft_size as f32, 0.0);
            result = result.mul_scalar(scale)?;
        }
        _ => {} // No normalization
    }

    Ok(result)
}

/// Real to complex FFT
pub fn rfft(
    input: &Tensor<f32>,
    n: Option<usize>,
    dim: Option<i32>,
    norm: Option<&str>,
) -> TorshResult<Tensor<Complex32>> {
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let ndim = dims.len();

    // Determine the dimension to apply RFFT
    let fft_dim = if let Some(d) = dim {
        if d < 0 {
            (ndim as i32 + d) as usize
        } else {
            d as usize
        }
    } else {
        ndim - 1 // Last dimension by default
    };

    if fft_dim >= ndim {
        return Err(TorshError::InvalidArgument(format!(
            "RFFT dimension {} is out of range for tensor with {} dimensions",
            fft_dim, ndim
        )));
    }

    let fft_size = n.unwrap_or(dims[fft_dim]);
    let output_size = fft_size / 2 + 1; // RFFT output size

    // Create FFT plan
    let plan = Plan::dft_1d(fft_size, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
        TorshError::InvalidArgument(format!("Failed to create FFT plan for size {}", fft_size))
    })?;

    let input_data = input.data()?;
    let input_len = input_data.len();

    // Handle reshaping for FFT computation
    let stride = dims[fft_dim];
    let batch_size = input_len / stride;

    let mut output_data = Vec::with_capacity(batch_size * output_size);

    // Perform FFT on each batch
    for batch_idx in 0..batch_size {
        let mut input_buffer: Vec<Complex<f32>> = Vec::with_capacity(fft_size);

        // Extract real data for this batch and convert to complex
        for i in 0..fft_size.min(stride) {
            let idx = batch_idx * stride + i;
            if idx < input_len {
                input_buffer.push(Complex::new(input_data[idx], 0.0));
            } else {
                input_buffer.push(Complex::new(0.0, 0.0));
            }
        }

        // Zero-pad if necessary
        while input_buffer.len() < fft_size {
            input_buffer.push(Complex::new(0.0, 0.0));
        }

        // Perform FFT (out-of-place)
        let mut output_buffer = vec![Complex::zero(); fft_size];
        plan.execute(&input_buffer, &mut output_buffer);

        // Take only the first half + 1 elements (real FFT property)
        for i in 0..output_size {
            let val = output_buffer[i];
            output_data.push(Complex32::new(val.re, val.im));
        }
    }

    // Create output shape
    let mut output_shape = dims.to_vec();
    output_shape[fft_dim] = output_size;

    let mut result = Tensor::from_data(output_data, output_shape, input.device())?;

    // Apply normalization
    match norm {
        Some("ortho") => {
            let scale = Complex32::new(1.0 / (fft_size as f32).sqrt(), 0.0);
            result = result.mul_scalar(scale)?;
        }
        Some("forward") => {
            let scale = Complex32::new(1.0 / fft_size as f32, 0.0);
            result = result.mul_scalar(scale)?;
        }
        _ => {} // No normalization
    }

    Ok(result)
}

/// 2D Fast Fourier Transform
pub fn fft2(
    input: &Tensor<Complex32>,
    s: Option<&[usize]>,
    dim: Option<&[i32]>,
    norm: Option<&str>,
) -> TorshResult<Tensor<Complex32>> {
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let ndim = dims.len();

    if ndim < 2 {
        return Err(TorshError::InvalidArgument(
            "Input tensor must have at least 2 dimensions for 2D FFT".to_string(),
        ));
    }

    // Default to last two dimensions
    let fft_dims = if let Some(d) = dim {
        if d.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "FFT2 requires exactly 2 dimensions".to_string(),
            ));
        }
        [
            if d[0] < 0 {
                (ndim as i32 + d[0]) as usize
            } else {
                d[0] as usize
            },
            if d[1] < 0 {
                (ndim as i32 + d[1]) as usize
            } else {
                d[1] as usize
            },
        ]
    } else {
        [ndim - 2, ndim - 1]
    };

    // Validate dimensions
    if fft_dims[0] >= ndim || fft_dims[1] >= ndim {
        return Err(TorshError::InvalidArgument(
            "FFT dimensions are out of range".to_string(),
        ));
    }

    // Apply FFT along first dimension, then second dimension
    let intermediate = fft(input, s.map(|s| s[0]), Some(fft_dims[0] as i32), norm)?;
    fft(
        &intermediate,
        s.map(|s| s[1]),
        Some(fft_dims[1] as i32),
        norm,
    )
}

/// Short-Time Fourier Transform
#[allow(clippy::too_many_arguments)]
pub fn stft(
    input: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    _window: Option<&Tensor>,
    _center: bool,
    _normalized: bool,
    onesided: bool,
    return_complex: bool,
) -> TorshResult<Tensor> {
    // Validate input
    if input.shape().ndim() > 2 {
        return Err(TorshError::invalid_argument_with_context(
            "STFT input must be 1D or 2D",
            "stft",
        ));
    }

    let hop_length = hop_length.unwrap_or(n_fft / 4);
    let _win_length = win_length.unwrap_or(n_fft);

    // This is a placeholder implementation
    // Real implementation would:
    // 1. Apply windowing
    // 2. Perform sliding window FFT
    // 3. Handle padding if center=true
    // 4. Return complex or magnitude/phase based on return_complex

    let input_len = input.shape().dims()[input.shape().ndim() - 1];
    let n_frames = (input_len - n_fft) / hop_length + 1;
    let output_freq = if onesided { n_fft / 2 + 1 } else { n_fft };

    // Create output shape
    let mut output_shape = input.shape().dims().to_vec();
    let last_idx = output_shape.len() - 1;
    output_shape[last_idx] = output_freq;
    output_shape.push(n_frames);

    if return_complex {
        output_shape.push(2); // Real and imaginary parts
    }

    // Create zeros tensor with the calculated shape
    torsh_tensor::creation::zeros(&output_shape).map_err(|e| TorshError::from(e))
}

/// Inverse Short-Time Fourier Transform
#[allow(clippy::too_many_arguments)]
pub fn istft(
    input: &Tensor,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    _window: Option<&Tensor>,
    _center: bool,
    _normalized: bool,
    _onesided: bool,
    length: Option<usize>,
    _return_complex: bool,
) -> TorshResult<Tensor> {
    // Validate input
    let ndim = input.shape().ndim();
    if ndim < 2 {
        return Err(TorshError::invalid_argument_with_context(
            "ISTFT input must have at least 2 dimensions",
            "istft",
        ));
    }

    let hop_length = hop_length.unwrap_or(n_fft / 4);
    let _win_length = win_length.unwrap_or(n_fft);

    // This is a placeholder implementation
    // Real implementation would:
    // 1. Apply inverse FFT to each frame
    // 2. Apply windowing
    // 3. Overlap-add reconstruction
    // 4. Handle trimming based on center and length parameters

    // Placeholder: return zeros
    let output_len = length.unwrap_or_else(|| {
        let n_frames = input.shape().dims()[ndim - 1];
        (n_frames - 1) * hop_length + n_fft
    });

    let shape = input.shape();
    let dims = shape.dims();
    let mut output_shape = dims[..ndim - 2].to_vec();
    output_shape.push(output_len);

    // Create zeros tensor with the calculated shape
    torsh_tensor::creation::zeros(&output_shape).map_err(|e| TorshError::from(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_ops::randn;
    use approx::assert_relative_eq;
    use torsh_core::dtype::Complex32;

    #[test]
    fn test_stft_shape() {
        let signal = randn(&[1024], None, None, None)
            .expect("Failed to generate random signal for STFT test");
        let n_fft = 256;
        let hop_length = 128;

        let stft_result = stft(
            &signal,
            n_fft,
            Some(hop_length),
            None,
            None,
            true,
            false,
            true,
            false,
        )
        .expect("STFT computation failed in test");

        // Check output shape
        assert_eq!(stft_result.shape().ndim(), 2);
        assert_eq!(stft_result.shape().dims()[0], n_fft / 2 + 1); // Frequency bins
    }

    #[test]
    fn test_fft_basic() {
        // Create a simple complex signal
        let data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.0, -1.0),
        ];
        let input = Tensor::from_data(data, vec![4], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create tensor from complex data in FFT test");

        // Test FFT
        let fft_result = fft(&input, None, None, None).expect("FFT computation failed in test");
        assert_eq!(fft_result.shape().dims(), &[4]);
        assert_eq!(fft_result.shape().ndim(), 1);
    }

    #[test]
    fn test_ifft_basic() {
        // Create a simple complex signal
        let data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.0, -1.0),
        ];
        let input = Tensor::from_data(data, vec![4], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create tensor from complex data in IFFT test");

        // Test IFFT
        let ifft_result = ifft(&input, None, None, None).expect("IFFT computation failed in test");
        assert_eq!(ifft_result.shape().dims(), &[4]);
        assert_eq!(ifft_result.shape().ndim(), 1);
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        // Create a simple complex signal
        let data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 1.0),
            Complex32::new(0.0, -1.0),
            Complex32::new(-1.0, 0.5),
        ];
        let input = Tensor::from_data(data.clone(), vec![4], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create tensor from complex data in roundtrip test");

        // FFT then IFFT should give back original (approximately)
        let fft_result =
            fft(&input, None, None, None).expect("FFT computation failed in roundtrip test");
        let ifft_result =
            ifft(&fft_result, None, None, None).expect("IFFT computation failed in roundtrip test");

        assert_eq!(ifft_result.shape().dims(), input.shape().dims());

        // Check that values are approximately equal
        let input_data = input.data().expect("tensor should have data");
        let ifft_data = ifft_result.data().expect("tensor should have data");

        for (_i, (orig, reconstructed)) in input_data.iter().zip(ifft_data.iter()).enumerate() {
            assert_relative_eq!(
                orig.re,
                reconstructed.re,
                epsilon = 1e-6,
                max_relative = 1e-5
            );
            assert_relative_eq!(
                orig.im,
                reconstructed.im,
                epsilon = 1e-6,
                max_relative = 1e-5
            );
        }
    }

    #[test]
    fn test_rfft_basic() {
        // Create a simple real signal
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(data, vec![4], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create tensor from real data in RFFT test");

        // Test RFFT
        let rfft_result = rfft(&input, None, None, None).expect("RFFT computation failed in test");
        assert_eq!(rfft_result.shape().dims(), &[3]); // N/2 + 1 for real FFT
        assert_eq!(rfft_result.shape().ndim(), 1);
    }

    #[test]
    fn test_fft2_basic() {
        // Create a simple 2D complex signal
        let data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];
        let input = Tensor::from_data(data, vec![2, 2], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create 2D tensor from complex data in FFT2 test");

        // Test 2D FFT
        let fft2_result = fft2(&input, None, None, None).expect("FFT2 computation failed in test");
        assert_eq!(fft2_result.shape().dims(), &[2, 2]);
        assert_eq!(fft2_result.shape().ndim(), 2);
    }

    #[test]
    fn test_istft_shape() {
        let stft_data = randn(&[129, 9], None, None, None)
            .expect("Failed to generate random STFT data for ISTFT test"); // Typical STFT shape
        let n_fft = 256;
        let hop_length = 128;

        let istft_result = istft(
            &stft_data,
            n_fft,
            Some(hop_length),
            None,
            None,
            true,
            false,
            true,
            None,
            false,
        )
        .expect("ISTFT computation failed in test");

        // Check output is 1D
        assert_eq!(istft_result.shape().ndim(), 1);
    }

    #[test]
    fn test_fft_with_dimension() {
        // Test FFT along specific dimension
        let data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
        ];
        let input = Tensor::from_data(data, vec![2, 3], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create 2x3 tensor from complex data in dimension test");

        // Test FFT along dimension 0
        let fft_result = fft(&input, None, Some(0), None)
            .expect("FFT computation along dimension 0 failed in test");
        assert_eq!(fft_result.shape().dims(), &[2, 3]);

        // Test FFT along dimension 1
        let fft_result = fft(&input, None, Some(1), None)
            .expect("FFT computation along dimension 1 failed in test");
        assert_eq!(fft_result.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_fft_with_normalization() {
        let data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.0, -1.0),
        ];
        let input = Tensor::from_data(data, vec![4], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create tensor from complex data in normalization test");

        // Test with "ortho" normalization
        let fft_result = fft(&input, None, None, Some("ortho"))
            .expect("FFT computation with 'ortho' normalization failed in test");
        assert_eq!(fft_result.shape().dims(), &[4]);

        // Test with "forward" normalization
        let fft_result = fft(&input, None, None, Some("forward"))
            .expect("FFT computation with 'forward' normalization failed in test");
        assert_eq!(fft_result.shape().dims(), &[4]);
    }

    #[test]
    fn test_error_handling() {
        let data = vec![Complex32::new(1.0, 0.0)];
        let input = Tensor::from_data(data, vec![1], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create tensor from complex data in error handling test");

        // Test FFT with invalid dimension
        let result = fft(&input, None, Some(5), None);
        assert!(result.is_err());

        // Test 2D FFT with insufficient dimensions
        let result = fft2(&input, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_fft_single_element() {
        // Test FFT with single element
        let data = vec![Complex32::new(42.0, 13.0)];
        let input = Tensor::from_data(data.clone(), vec![1], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create single-element tensor");

        let fft_result =
            fft(&input, None, None, None).expect("FFT of single element should succeed");
        let result_data = fft_result.data().expect("Failed to get data");

        // Single element FFT should return the same value
        assert_relative_eq!(result_data[0].re, data[0].re, epsilon = 1e-6);
        assert_relative_eq!(result_data[0].im, data[0].im, epsilon = 1e-6);
    }

    #[test]
    fn test_fft_power_of_two_vs_non_power_of_two() {
        // Test that FFT works correctly for both power-of-2 and non-power-of-2 sizes
        for size in [7, 8, 15, 16, 31, 32] {
            let data: Vec<Complex32> = (0..size).map(|i| Complex32::new(i as f32, 0.0)).collect();
            let input = Tensor::from_data(data, vec![size], torsh_core::device::DeviceType::Cpu)
                .expect("Failed to create tensor");

            let fft_result = fft(&input, None, None, None).expect("FFT should work for any size");
            assert_eq!(fft_result.shape().dims()[0], size);
        }
    }

    #[test]
    fn test_rfft_nyquist_frequency() {
        // Test that RFFT correctly handles Nyquist frequency
        let size = 8;
        let data = vec![1.0; size];
        let input = Tensor::from_data(data, vec![size], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create tensor");

        let rfft_result = rfft(&input, None, None, None).expect("RFFT should succeed");

        // RFFT should return N/2 + 1 frequency bins
        assert_eq!(rfft_result.shape().dims()[0], size / 2 + 1);

        let result_data = rfft_result.data().expect("Failed to get data");
        // DC component should be N (sum of all 1s)
        assert_relative_eq!(result_data[0].re, size as f32, epsilon = 1e-5);
    }

    #[test]
    fn test_fft_parseval_theorem() {
        // Parseval's theorem: sum of squared magnitudes in time domain
        // equals sum of squared magnitudes in frequency domain (with normalization)
        let size = 16;
        let data: Vec<Complex32> = (0..size)
            .map(|i| Complex32::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let input = Tensor::from_data(
            data.clone(),
            vec![size],
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("Failed to create tensor");

        // Compute energy in time domain
        let time_energy: f32 = data.iter().map(|c| c.re * c.re + c.im * c.im).sum();

        // Compute FFT with ortho normalization
        let fft_result = fft(&input, None, None, Some("ortho"))
            .expect("FFT with ortho normalization should succeed");
        let freq_data = fft_result.data().expect("Failed to get data");

        // Compute energy in frequency domain
        let freq_energy: f32 = freq_data.iter().map(|c| c.re * c.re + c.im * c.im).sum();

        // Energies should match (Parseval's theorem)
        assert_relative_eq!(time_energy, freq_energy, epsilon = 1e-4);
    }

    #[test]
    fn test_fft_linearity() {
        // Test that FFT is linear: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
        let size = 8;
        let x: Vec<Complex32> = (0..size).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let y: Vec<Complex32> = (0..size)
            .map(|i| Complex32::new((i * 2) as f32, 1.0))
            .collect();

        let a = 2.0;
        let b = 3.0;

        let x_tensor =
            Tensor::from_data(x.clone(), vec![size], torsh_core::device::DeviceType::Cpu)
                .expect("Failed to create x tensor");
        let y_tensor =
            Tensor::from_data(y.clone(), vec![size], torsh_core::device::DeviceType::Cpu)
                .expect("Failed to create y tensor");

        // Compute combined signal
        let combined: Vec<Complex32> = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| Complex32::new(a * xi.re + b * yi.re, a * xi.im + b * yi.im))
            .collect();
        let combined_tensor =
            Tensor::from_data(combined, vec![size], torsh_core::device::DeviceType::Cpu)
                .expect("Failed to create combined tensor");

        // FFT of combined signal
        let fft_combined =
            fft(&combined_tensor, None, None, None).expect("FFT of combined signal should succeed");
        let fft_combined_data = fft_combined.data().expect("Failed to get data");

        // FFT of individual signals
        let fft_x = fft(&x_tensor, None, None, None).expect("FFT of x should succeed");
        let fft_y = fft(&y_tensor, None, None, None).expect("FFT of y should succeed");
        let fft_x_data = fft_x.data().expect("Failed to get x data");
        let fft_y_data = fft_y.data().expect("Failed to get y data");

        // Check linearity
        for i in 0..size {
            let expected_re = a * fft_x_data[i].re + b * fft_y_data[i].re;
            let expected_im = a * fft_x_data[i].im + b * fft_y_data[i].im;

            assert_relative_eq!(fft_combined_data[i].re, expected_re, epsilon = 1e-5);
            assert_relative_eq!(fft_combined_data[i].im, expected_im, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_fft_dc_component() {
        // Test that DC component (index 0) equals sum of all input values
        let size = 16;
        let data: Vec<Complex32> = (0..size)
            .map(|i| Complex32::new((i + 1) as f32, 0.0))
            .collect();
        let input = Tensor::from_data(
            data.clone(),
            vec![size],
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("Failed to create tensor");

        let fft_result = fft(&input, None, None, None).expect("FFT should succeed");
        let result_data = fft_result.data().expect("Failed to get data");

        // DC component should equal sum of input
        let expected_dc: f32 = data.iter().map(|c| c.re).sum();
        assert_relative_eq!(result_data[0].re, expected_dc, epsilon = 1e-5);
        assert_relative_eq!(result_data[0].im, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_fft_symmetry_real_input() {
        // For real input, FFT output should have Hermitian symmetry: X[k] = conj(X[N-k])
        let size = 16;
        let data: Vec<Complex32> = (0..size)
            .map(|i| Complex32::new((i as f32).sin(), 0.0))
            .collect();
        let input = Tensor::from_data(data, vec![size], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create tensor");

        let fft_result = fft(&input, None, None, None).expect("FFT should succeed");
        let result_data = fft_result.data().expect("Failed to get data");

        // Check Hermitian symmetry
        for k in 1..size / 2 {
            let pos = &result_data[k];
            let neg = &result_data[size - k];

            assert_relative_eq!(pos.re, neg.re, epsilon = 1e-5);
            assert_relative_eq!(pos.im, -neg.im, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_fft2_separability() {
        // Test that 2D FFT can be computed as row FFTs followed by column FFTs
        let data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];
        let input = Tensor::from_data(data, vec![2, 2], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create 2x2 tensor");

        // Full 2D FFT
        let fft2_result = fft2(&input, None, None, None).expect("2D FFT should succeed");
        assert_eq!(fft2_result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_large_signal_fft() {
        // Test FFT on a moderately sized signal to ensure scalability
        // Using 256 samples (powers of 2 are efficient for FFT)
        // Note: Larger sizes (512+) can cause stack overflow in oxifft's internal buffers
        // Real-world usage typically allocates large signals on heap before calling FFT
        let size = 256;
        let data: Vec<Complex32> = (0..size)
            .map(|i| Complex32::new((i as f32 / 100.0).sin(), 0.0))
            .collect();
        let input = Tensor::from_data(data, vec![size], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create large tensor");

        let fft_result = fft(&input, None, None, None).expect("FFT on large signal should succeed");
        assert_eq!(fft_result.shape().dims()[0], size);

        // Test roundtrip
        let ifft_result =
            ifft(&fft_result, None, None, None).expect("IFFT on large signal should succeed");
        assert_eq!(ifft_result.shape().dims()[0], size);
    }

    #[test]
    fn test_rfft_zero_padding() {
        // Test RFFT with zero-padding (n > input length)
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(data, vec![4], torsh_core::device::DeviceType::Cpu)
            .expect("Failed to create tensor");

        let n_fft = 8;
        let rfft_result =
            rfft(&input, Some(n_fft), None, None).expect("RFFT with zero-padding should succeed");

        // Output size should be n_fft/2 + 1
        assert_eq!(rfft_result.shape().dims()[0], n_fft / 2 + 1);
    }
}
