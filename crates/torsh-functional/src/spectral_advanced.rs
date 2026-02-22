//! Advanced spectral operations - Extended FFT variants and spectral analysis
//!
//! This module provides advanced FFT operations including:
//! - N-dimensional FFT (fftn, ifftn)
//! - Inverse real FFT (irfft)
//! - Multi-dimensional real FFT (rfft2, rfftn)
//! - Hermitian FFT (hfft, ihfft)
//! - Complete STFT/ISTFT with windowing
//! - Spectral analysis functions

use torsh_core::{dtype::Complex32, Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

use crate::spectral::{fft, ifft, rfft};

/// N-dimensional Fast Fourier Transform
///
/// Computes the N-dimensional discrete Fourier Transform over any number of axes in an M-dimensional array.
///
/// # Arguments
///
/// * `input` - Input complex tensor
/// * `s` - Signal size for each transformed dimension (None uses input size)
/// * `dim` - Dimensions along which to apply FFT (None uses all dimensions)
/// * `norm` - Normalization mode: None (no normalization), "ortho" (1/sqrt(N)), or "forward" (1/N)
///
/// # Mathematical Formula
///
/// For each dimension k, applies:
/// ```text
/// X[m] = Σ(n=0 to N-1) x[n] * exp(-2πi * m * n / N)
/// ```
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_functional::spectral_advanced::fftn;
/// use torsh_core::dtype::Complex32;
/// use torsh_tensor::Tensor;
///
/// let data = vec![Complex32::new(1.0, 0.0); 64];
/// let input = Tensor::from_data(data, vec![4, 4, 4], DeviceType::Cpu)?;
/// let result = fftn(&input, None, None, Some("ortho"))?;
/// ```
pub fn fftn(
    input: &Tensor<Complex32>,
    s: Option<&[usize]>,
    dim: Option<&[i32]>,
    norm: Option<&str>,
) -> TorshResult<Tensor<Complex32>> {
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let ndim = dims.len();

    // Determine which dimensions to apply FFT
    let fft_dims: Vec<usize> = if let Some(d) = dim {
        d.iter()
            .map(|&x| {
                if x < 0 {
                    (ndim as i32 + x) as usize
                } else {
                    x as usize
                }
            })
            .collect()
    } else {
        (0..ndim).collect()
    };

    // Validate dimensions
    for &d in &fft_dims {
        if d >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "FFT dimension {} is out of range for tensor with {} dimensions",
                d, ndim
            )));
        }
    }

    // Apply FFT sequentially along each dimension
    let mut result = input.clone();
    for (idx, &dim_idx) in fft_dims.iter().enumerate() {
        let size = if let Some(sizes) = s {
            if idx < sizes.len() {
                Some(sizes[idx])
            } else {
                None
            }
        } else {
            None
        };
        result = fft(&result, size, Some(dim_idx as i32), norm)?;
    }

    Ok(result)
}

/// Inverse N-dimensional Fast Fourier Transform
///
/// Computes the inverse of the N-dimensional discrete Fourier Transform.
///
/// # Arguments
///
/// * `input` - Input complex tensor
/// * `s` - Signal size for each transformed dimension (None uses input size)
/// * `dim` - Dimensions along which to apply IFFT (None uses all dimensions)
/// * `norm` - Normalization mode: None or "backward" (1/N), "ortho" (1/sqrt(N)), or "forward" (no normalization)
///
/// # Examples
///
/// ```rust,ignore
/// let fft_result = fftn(&input, None, None, None)?;
/// let recovered = ifftn(&fft_result, None, None, None)?;
/// // recovered ≈ input
/// ```
pub fn ifftn(
    input: &Tensor<Complex32>,
    s: Option<&[usize]>,
    dim: Option<&[i32]>,
    norm: Option<&str>,
) -> TorshResult<Tensor<Complex32>> {
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let ndim = dims.len();

    // Determine which dimensions to apply IFFT
    let fft_dims: Vec<usize> = if let Some(d) = dim {
        d.iter()
            .map(|&x| {
                if x < 0 {
                    (ndim as i32 + x) as usize
                } else {
                    x as usize
                }
            })
            .collect()
    } else {
        (0..ndim).collect()
    };

    // Validate dimensions
    for &d in &fft_dims {
        if d >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "IFFT dimension {} is out of range for tensor with {} dimensions",
                d, ndim
            )));
        }
    }

    // Apply IFFT sequentially along each dimension
    let mut result = input.clone();
    for (idx, &dim_idx) in fft_dims.iter().enumerate() {
        let size = if let Some(sizes) = s {
            if idx < sizes.len() {
                Some(sizes[idx])
            } else {
                None
            }
        } else {
            None
        };
        result = ifft(&result, size, Some(dim_idx as i32), norm)?;
    }

    Ok(result)
}

/// Inverse Real FFT
///
/// Computes the inverse of `rfft`, returning a real-valued tensor.
/// Assumes input has Hermitian symmetry (conjugate symmetry).
///
/// # Arguments
///
/// * `input` - Input complex tensor from RFFT
/// * `n` - Output signal size (None infers from input: (input_size - 1) * 2)
/// * `dim` - Dimension along which to apply IRFFT
/// * `norm` - Normalization mode
///
/// # Mathematical Formula
///
/// Recovers real signal from Hermitian-symmetric frequency representation:
/// ```text
/// x[n] = IFFT(X[k]) where X[k] = X*[N-k] (Hermitian symmetry)
/// ```
///
/// # Examples
///
/// ```rust,ignore
/// let real_signal = randn(&[1024], None, None, None)?;
/// let freq = rfft(&real_signal, None, None, None)?;
/// let recovered = irfft(&freq, Some(1024), None, None)?;
/// // recovered ≈ real_signal
/// ```
pub fn irfft(
    input: &Tensor<Complex32>,
    n: Option<usize>,
    dim: Option<i32>,
    norm: Option<&str>,
) -> TorshResult<Tensor<f32>> {
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let ndim = dims.len();

    // Determine the dimension to apply IRFFT
    let fft_dim = if let Some(d) = dim {
        if d < 0 {
            (ndim as i32 + d) as usize
        } else {
            d as usize
        }
    } else {
        ndim - 1
    };

    if fft_dim >= ndim {
        return Err(TorshError::InvalidArgument(format!(
            "IRFFT dimension {} is out of range for tensor with {} dimensions",
            fft_dim, ndim
        )));
    }

    // Infer output size from input if not provided
    let output_size = n.unwrap_or((dims[fft_dim] - 1) * 2);

    // Create full complex FFT input by mirroring (Hermitian symmetry)
    let input_data = input.data()?;
    let stride = dims[fft_dim];
    let batch_size = input_data.len() / stride;

    let mut full_fft_data = Vec::with_capacity(batch_size * output_size);

    for batch_idx in 0..batch_size {
        let mut batch_data = Vec::with_capacity(output_size);

        // Copy positive frequencies
        for i in 0..stride.min(output_size) {
            let idx = batch_idx * stride + i;
            if idx < input_data.len() {
                batch_data.push(input_data[idx]);
            } else {
                batch_data.push(Complex32::new(0.0, 0.0));
            }
        }

        // Add mirrored negative frequencies (conjugate symmetry for real signals)
        for i in stride..(output_size) {
            let mirror_idx = output_size - i;
            if mirror_idx < batch_data.len() && mirror_idx > 0 {
                let val = batch_data[mirror_idx];
                batch_data.push(Complex32::new(val.re, -val.im)); // Complex conjugate
            } else {
                batch_data.push(Complex32::new(0.0, 0.0));
            }
        }

        full_fft_data.extend(batch_data);
    }

    // Create complex tensor with full FFT data
    let mut full_shape = dims.to_vec();
    full_shape[fft_dim] = output_size;
    let complex_tensor = Tensor::from_data(full_fft_data, full_shape, input.device())?;

    // Apply IFFT
    let ifft_result = ifft(
        &complex_tensor,
        Some(output_size),
        Some(fft_dim as i32),
        norm,
    )?;

    // Extract real part
    let ifft_data = ifft_result.data()?;
    let real_data: Vec<f32> = ifft_data.iter().map(|c| c.re).collect();

    // Create output tensor
    let output_shape = ifft_result.shape().dims().to_vec();
    Tensor::from_data(real_data, output_shape, input.device())
}

/// 2D Real to Complex FFT
///
/// Computes 2D FFT of real input, exploiting Hermitian symmetry in the last dimension.
///
/// # Arguments
///
/// * `input` - Input real tensor
/// * `s` - Signal sizes for each dimension [rows, cols]
/// * `dim` - Dimensions along which to apply FFT [dim0, dim1]
/// * `norm` - Normalization mode
///
/// # Complexity
///
/// - Time: O(N * M * log(N * M)) where N×M is the 2D signal size
/// - Space: O(N * (M/2 + 1)) due to Hermitian symmetry
pub fn rfft2(
    input: &Tensor<f32>,
    s: Option<&[usize]>,
    dim: Option<&[i32]>,
    norm: Option<&str>,
) -> TorshResult<Tensor<Complex32>> {
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let ndim = dims.len();

    if ndim < 2 {
        return Err(TorshError::InvalidArgument(
            "Input tensor must have at least 2 dimensions for 2D RFFT".to_string(),
        ));
    }

    // Default to last two dimensions
    let fft_dims = if let Some(d) = dim {
        if d.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "RFFT2 requires exactly 2 dimensions".to_string(),
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

    // First, convert real tensor to complex
    let input_data = input.data()?;
    let complex_data: Vec<Complex32> = input_data.iter().map(|&x| Complex32::new(x, 0.0)).collect();
    let complex_input = Tensor::from_data(complex_data, dims.to_vec(), input.device())?;

    // Apply FFT along first dimension
    let intermediate = fft(
        &complex_input,
        s.map(|s| s[0]),
        Some(fft_dims[0] as i32),
        norm,
    )?;

    // Apply RFFT along second dimension (only half the frequencies needed)
    // Extract real part for RFFT
    let inter_data = intermediate.data()?;
    let real_inter: Vec<f32> = inter_data.iter().map(|c| c.re).collect();
    let real_intermediate = Tensor::from_data(
        real_inter,
        intermediate.shape().dims().to_vec(),
        intermediate.device(),
    )?;

    rfft(
        &real_intermediate,
        s.map(|s| s[1]),
        Some(fft_dims[1] as i32),
        norm,
    )
}

/// N-dimensional Real to Complex FFT
///
/// Computes N-dimensional FFT of real input, exploiting Hermitian symmetry in the last dimension.
///
/// # Arguments
///
/// * `input` - Input real tensor
/// * `s` - Signal sizes for each dimension
/// * `dim` - Dimensions along which to apply FFT
/// * `norm` - Normalization mode
///
/// # Examples
///
/// ```rust,ignore
/// let real_data = randn(&[32, 32, 32], None, None, None)?;
/// let freq = rfftn(&real_data, None, None, None)?;
/// // freq.shape() == [32, 32, 17] (last dim is N/2 + 1)
/// ```
pub fn rfftn(
    input: &Tensor<f32>,
    s: Option<&[usize]>,
    dim: Option<&[i32]>,
    norm: Option<&str>,
) -> TorshResult<Tensor<Complex32>> {
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let ndim = dims.len();

    // Determine which dimensions to apply FFT
    let fft_dims: Vec<usize> = if let Some(d) = dim {
        d.iter()
            .map(|&x| {
                if x < 0 {
                    (ndim as i32 + x) as usize
                } else {
                    x as usize
                }
            })
            .collect()
    } else {
        (0..ndim).collect()
    };

    // Validate dimensions
    for &d in &fft_dims {
        if d >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "FFT dimension {} is out of range for tensor with {} dimensions",
                d, ndim
            )));
        }
    }

    if fft_dims.is_empty() {
        return Err(TorshError::InvalidArgument(
            "At least one dimension required for RFFTN".to_string(),
        ));
    }

    // Convert real input to complex
    let input_data = input.data()?;
    let complex_data: Vec<Complex32> = input_data.iter().map(|&x| Complex32::new(x, 0.0)).collect();
    let mut result = Tensor::from_data(complex_data, dims.to_vec(), input.device())?;

    // Apply FFT along all dimensions except the last
    for (idx, &dim_idx) in fft_dims.iter().enumerate() {
        if idx < fft_dims.len() - 1 {
            let size = if let Some(sizes) = s {
                if idx < sizes.len() {
                    Some(sizes[idx])
                } else {
                    None
                }
            } else {
                None
            };
            result = fft(&result, size, Some(dim_idx as i32), norm)?;
        }
    }

    // Apply RFFT on the last dimension
    let last_dim = fft_dims[fft_dims.len() - 1];
    let last_size = if let Some(sizes) = s {
        if fft_dims.len() - 1 < sizes.len() {
            Some(sizes[fft_dims.len() - 1])
        } else {
            None
        }
    } else {
        None
    };

    // Extract real part for RFFT
    let result_data = result.data()?;
    let real_data: Vec<f32> = result_data.iter().map(|c| c.re).collect();
    let real_result =
        Tensor::from_data(real_data, result.shape().dims().to_vec(), result.device())?;

    rfft(&real_result, last_size, Some(last_dim as i32), norm)
}

/// Hermitian FFT
///
/// Computes the FFT of a signal with Hermitian symmetry (real spectrum).
/// Input is assumed to have Hermitian symmetry in time domain, output is real-valued in frequency domain.
///
/// # Arguments
///
/// * `input` - Input complex tensor with Hermitian symmetry
/// * `n` - Output size
/// * `dim` - Dimension along which to apply HFFT
/// * `norm` - Normalization mode
///
/// # Notes
///
/// This is the opposite of `rfft`: `hfft(rfft(x)) ≈ x` for real signals.
pub fn hfft(
    input: &Tensor<Complex32>,
    n: Option<usize>,
    dim: Option<i32>,
    norm: Option<&str>,
) -> TorshResult<Tensor<f32>> {
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let ndim = dims.len();

    let fft_dim = if let Some(d) = dim {
        if d < 0 {
            (ndim as i32 + d) as usize
        } else {
            d as usize
        }
    } else {
        ndim - 1
    };

    if fft_dim >= ndim {
        return Err(TorshError::InvalidArgument(format!(
            "HFFT dimension {} is out of range",
            fft_dim
        )));
    }

    let output_size = n.unwrap_or((dims[fft_dim] - 1) * 2);

    // Build full complex signal with Hermitian symmetry
    let input_data = input.data()?;
    let stride = dims[fft_dim];
    let batch_size = input_data.len() / stride;

    let mut full_data = Vec::with_capacity(batch_size * output_size);

    for batch_idx in 0..batch_size {
        let mut batch = Vec::with_capacity(output_size);

        // Copy first half
        for i in 0..stride.min(output_size) {
            let idx = batch_idx * stride + i;
            if idx < input_data.len() {
                batch.push(input_data[idx]);
            } else {
                batch.push(Complex32::new(0.0, 0.0));
            }
        }

        // Mirror with conjugate
        for i in stride..output_size {
            let mirror = output_size - i;
            if mirror < batch.len() && mirror > 0 {
                let v = batch[mirror];
                batch.push(Complex32::new(v.re, -v.im));
            } else {
                batch.push(Complex32::new(0.0, 0.0));
            }
        }

        full_data.extend(batch);
    }

    let mut shape = dims.to_vec();
    shape[fft_dim] = output_size;
    let full_tensor = Tensor::from_data(full_data, shape, input.device())?;

    // Apply regular FFT
    let fft_result = fft(&full_tensor, Some(output_size), Some(fft_dim as i32), norm)?;

    // Extract real part (should be nearly real due to Hermitian input)
    let fft_data = fft_result.data()?;
    let real: Vec<f32> = fft_data.iter().map(|c| c.re).collect();

    Tensor::from_data(
        real,
        fft_result.shape().dims().to_vec(),
        fft_result.device(),
    )
}

/// Inverse Hermitian FFT
///
/// Computes the inverse of `hfft`, returning a complex tensor with Hermitian symmetry.
///
/// # Arguments
///
/// * `input` - Input real tensor
/// * `n` - Output size
/// * `dim` - Dimension along which to apply IHFFT
/// * `norm` - Normalization mode
///
/// # Notes
///
/// The output will have Hermitian symmetry: `output[k] = conj(output[n-k])`
pub fn ihfft(
    input: &Tensor<f32>,
    n: Option<usize>,
    dim: Option<i32>,
    norm: Option<&str>,
) -> TorshResult<Tensor<Complex32>> {
    // IHFFT is essentially IRFFT
    // Convert to complex, apply IFFT, then extract Hermitian part
    let input_data = input.data()?;
    let complex_data: Vec<Complex32> = input_data.iter().map(|&x| Complex32::new(x, 0.0)).collect();

    let input_shape = input.shape();
    let dims = input_shape.dims();
    let complex_input = Tensor::from_data(complex_data, dims.to_vec(), input.device())?;

    let fft_dim = dim.unwrap_or(-1);
    let ifft_result = ifft(&complex_input, n, Some(fft_dim), norm)?;

    // Extract Hermitian part (first half + DC)
    let ifft_data = ifft_result.data()?;
    let shape_obj = ifft_result.shape();
    let ifft_shape = shape_obj.dims();
    let ndim = ifft_shape.len();

    let target_dim = if fft_dim < 0 {
        (ndim as i32 + fft_dim) as usize
    } else {
        fft_dim as usize
    };

    let hermitian_size = ifft_shape[target_dim] / 2 + 1;
    let stride = ifft_shape[target_dim];
    let batch_size = ifft_data.len() / stride;

    let mut hermitian_data = Vec::with_capacity(batch_size * hermitian_size);
    for batch_idx in 0..batch_size {
        for i in 0..hermitian_size {
            let idx = batch_idx * stride + i;
            if idx < ifft_data.len() {
                hermitian_data.push(ifft_data[idx]);
            } else {
                hermitian_data.push(Complex32::new(0.0, 0.0));
            }
        }
    }

    let mut output_shape = ifft_shape.to_vec();
    output_shape[target_dim] = hermitian_size;

    Tensor::from_data(hermitian_data, output_shape, ifft_result.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_fftn_basic() -> TorshResult<()> {
        let data = vec![Complex32::new(1.0, 0.0); 8];
        let input = Tensor::from_data(data, vec![2, 2, 2], DeviceType::Cpu)?;

        let result = fftn(&input, None, None, None)?;
        assert_eq!(result.shape().dims(), &[2, 2, 2]);
        Ok(())
    }

    #[test]
    fn test_fftn_ifftn_roundtrip() -> TorshResult<()> {
        let data = vec![
            Complex32::new(1.0, 0.5),
            Complex32::new(2.0, 1.0),
            Complex32::new(0.5, -1.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.0, 1.5),
            Complex32::new(1.5, 0.5),
            Complex32::new(-0.5, -0.5),
            Complex32::new(0.5, -1.5),
        ];
        let input = Tensor::from_data(data.clone(), vec![2, 2, 2], DeviceType::Cpu)?;

        let fft_result = fftn(&input, None, None, None)?;
        let ifft_result = ifftn(&fft_result, None, None, None)?;

        let input_data = input.data()?;
        let result_data = ifft_result.data()?;

        for (orig, recovered) in input_data.iter().zip(result_data.iter()) {
            assert_relative_eq!(orig.re, recovered.re, epsilon = 1e-5);
            assert_relative_eq!(orig.im, recovered.im, epsilon = 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_irfft_basic() -> TorshResult<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(data, vec![4], DeviceType::Cpu)?;

        let rfft_result = rfft(&input, None, None, None)?;
        assert_eq!(rfft_result.shape().dims(), &[3]); // N/2 + 1

        let irfft_result = irfft(&rfft_result, Some(4), None, None)?;
        assert_eq!(irfft_result.shape().dims(), &[4]);
        Ok(())
    }

    #[test]
    fn test_rfft_irfft_roundtrip() -> TorshResult<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Tensor::from_data(data.clone(), vec![8], DeviceType::Cpu)?;

        let rfft_result = rfft(&input, None, None, None)?;
        let irfft_result = irfft(&rfft_result, Some(8), None, None)?;

        let input_data = input.data()?;
        let result_data = irfft_result.data()?;

        for (orig, recovered) in input_data.iter().zip(result_data.iter()) {
            assert_relative_eq!(orig, recovered, epsilon = 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_rfft2_basic() -> TorshResult<()> {
        let data = vec![1.0; 16];
        let input = Tensor::from_data(data, vec![4, 4], DeviceType::Cpu)?;

        let result = rfft2(&input, None, None, None)?;
        // Last dimension should be N/2 + 1 = 3
        assert_eq!(result.shape().dims()[1], 3);
        Ok(())
    }

    #[test]
    fn test_rfftn_basic() -> TorshResult<()> {
        let data = vec![1.0; 64];
        let input = Tensor::from_data(data, vec![4, 4, 4], DeviceType::Cpu)?;

        let result = rfftn(&input, None, None, None)?;
        // Last dimension should be N/2 + 1 = 3
        let result_shape_obj = result.shape();
        let result_shape = result_shape_obj.dims();
        assert_eq!(result_shape[2], 3);
        Ok(())
    }

    #[test]
    fn test_hfft_basic() -> TorshResult<()> {
        let data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 1.0),
            Complex32::new(3.0, 0.5),
        ];
        let input = Tensor::from_data(data, vec![3], DeviceType::Cpu)?;

        let result = hfft(&input, Some(4), None, None)?;
        assert_eq!(result.shape().dims(), &[4]);
        Ok(())
    }

    #[test]
    fn test_ihfft_basic() -> TorshResult<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(data, vec![4], DeviceType::Cpu)?;

        let result = ihfft(&input, Some(4), None, None)?;
        // Result should have Hermitian symmetry
        assert_eq!(result.shape().dims(), &[3]); // N/2 + 1
        Ok(())
    }

    #[test]
    fn test_fftn_with_specific_dims() -> TorshResult<()> {
        let data = vec![Complex32::new(1.0, 0.0); 24];
        let input = Tensor::from_data(data, vec![2, 3, 4], DeviceType::Cpu)?;

        // Apply FFT only on dimensions 0 and 2
        let result = fftn(&input, None, Some(&[0, 2]), None)?;
        assert_eq!(result.shape().dims(), &[2, 3, 4]);
        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let data = vec![Complex32::new(1.0, 0.0)];
        let input = Tensor::from_data(data, vec![1], DeviceType::Cpu).unwrap();

        // Test with invalid dimension
        let result = fftn(&input, None, Some(&[5]), None);
        assert!(result.is_err());

        // Test rfftn with no dimensions
        let real_data = vec![1.0];
        let real_input = Tensor::from_data(real_data, vec![1], DeviceType::Cpu).unwrap();
        let result = rfftn(&real_input, None, Some(&[]), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_fftn_3d_cube() -> TorshResult<()> {
        // Test 3D FFT on a cube
        let size = 8;
        let data = vec![Complex32::new(1.0, 0.0); size * size * size];
        let input = Tensor::from_data(data, vec![size, size, size], DeviceType::Cpu)?;

        let result = fftn(&input, None, None, None)?;
        let result_shape = result.shape();
        assert_eq!(result_shape.dims(), &[size, size, size]);

        // Test roundtrip
        let recovered = ifftn(&result, None, None, None)?;
        assert_eq!(recovered.shape().dims(), &[size, size, size]);
        Ok(())
    }

    #[test]
    fn test_irfft_odd_even_sizes() -> TorshResult<()> {
        // Test IRFFT with both odd and even output sizes
        for n in [7, 8, 15, 16] {
            let data = vec![1.0, 2.0, 3.0, 4.0];
            let input = Tensor::from_data(data, vec![4], DeviceType::Cpu)?;

            let rfft_result = rfft(&input, Some(n), None, None)?;
            let irfft_result = irfft(&rfft_result, Some(n), None, None)?;

            assert_eq!(irfft_result.shape().dims()[0], n);
        }
        Ok(())
    }

    #[test]
    fn test_rfft2_rectangular() -> TorshResult<()> {
        // Test 2D RFFT on rectangular (non-square) arrays
        let data = vec![1.0; 12];
        let input = Tensor::from_data(data, vec![3, 4], DeviceType::Cpu)?;

        let result = rfft2(&input, None, None, None)?;
        let result_shape = result.shape();
        let dims = result_shape.dims();

        // Last dimension should be N/2 + 1
        assert_eq!(dims[1], 3); // 4/2 + 1 = 3
        Ok(())
    }

    #[test]
    fn test_rfftn_preserves_energy() -> TorshResult<()> {
        // Test that RFFTN preserves total energy (Parseval's theorem for real signals)
        let size = 16;
        let data: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
        let input = Tensor::from_data(data.clone(), vec![size], DeviceType::Cpu)?;

        // Time domain energy
        let time_energy: f32 = data.iter().map(|&x| x * x).sum();

        // Frequency domain energy (with ortho normalization)
        let freq = rfftn(&input, None, None, Some("ortho"))?;
        let freq_data = freq.data()?;

        // For RFFT, need to account for symmetric spectrum
        let mut freq_energy = 0.0f32;
        for (i, c) in freq_data.iter().enumerate() {
            let mag_sq = c.re * c.re + c.im * c.im;
            if i == 0 || i == freq_data.len() - 1 {
                freq_energy += mag_sq;
            } else {
                freq_energy += 2.0 * mag_sq; // Count symmetric part
            }
        }

        assert_relative_eq!(time_energy, freq_energy, epsilon = 0.1);
        Ok(())
    }

    #[test]
    fn test_hfft_ihfft_consistency() -> TorshResult<()> {
        // Test that HFFT and IHFFT are consistent
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Tensor::from_data(data, vec![8], DeviceType::Cpu)?;

        let hermitian = ihfft(&input, Some(8), None, None)?;
        let recovered = hfft(&hermitian, Some(8), None, None)?;

        let input_data = input.data()?;
        let recovered_data = recovered.data()?;

        // Allow some tolerance due to numerical errors
        for i in 0..input_data.len().min(recovered_data.len()) {
            assert_relative_eq!(input_data[i], recovered_data[i], epsilon = 1e-3);
        }
        Ok(())
    }

    #[test]
    fn test_fftn_normalization_modes() -> TorshResult<()> {
        // Test different normalization modes for N-D FFT
        let data = vec![Complex32::new(1.0, 0.0); 27];
        let input = Tensor::from_data(data, vec![3, 3, 3], DeviceType::Cpu)?;

        for norm in [None, Some("ortho"), Some("forward")] {
            let result = fftn(&input, None, None, norm)?;
            assert_eq!(result.shape().dims(), &[3, 3, 3]);

            // Test roundtrip
            let recovered = ifftn(&result, None, None, norm)?;
            let input_data = input.data()?;
            let recovered_data = recovered.data()?;

            for i in 0..input_data.len() {
                assert_relative_eq!(input_data[i].re, recovered_data[i].re, epsilon = 1e-4);
                assert_relative_eq!(input_data[i].im, recovered_data[i].im, epsilon = 1e-4);
            }
        }
        Ok(())
    }

    #[test]
    fn test_rfft2_on_tall_and_wide_matrices() -> TorshResult<()> {
        // Test RFFT2 on tall (more rows) and wide (more columns) matrices
        let tall_data = vec![1.0; 32]; // 8x4
        let tall = Tensor::from_data(tall_data, vec![8, 4], DeviceType::Cpu)?;
        let tall_result = rfft2(&tall, None, None, None)?;
        assert!(tall_result.shape().dims()[1] <= 3); // 4/2 + 1

        let wide_data = vec![1.0; 24]; // 3x8
        let wide = Tensor::from_data(wide_data, vec![3, 8], DeviceType::Cpu)?;
        let wide_result = rfft2(&wide, None, None, None)?;
        assert!(wide_result.shape().dims()[1] <= 5); // 8/2 + 1

        Ok(())
    }

    #[test]
    fn test_irfft_length_inference() -> TorshResult<()> {
        // Test that IRFFT correctly infers output length when not specified
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Tensor::from_data(data, vec![8], DeviceType::Cpu)?;

        let rfft_result = rfft(&input, None, None, None)?;
        // IRFFT should infer length from input shape
        let irfft_result = irfft(&rfft_result, None, None, None)?;

        // Inferred length should be (n_freq - 1) * 2 = (5 - 1) * 2 = 8
        assert_eq!(irfft_result.shape().dims()[0], 8);
        Ok(())
    }
}
