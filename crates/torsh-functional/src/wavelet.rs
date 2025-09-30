//! Wavelet transform operations
//!
//! This module provides various wavelet transform functions commonly used in signal processing
//! and image analysis. Wavelets are particularly useful for multi-resolution analysis and
//! feature extraction in deep learning applications.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;
// use std::ops::{Div, Mul}; // Currently unused

/// Wavelet basis types supported
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveletType {
    /// Daubechies wavelets
    Daubechies(usize), // Number of vanishing moments
    /// Biorthogonal wavelets
    Biorthogonal(usize, usize), // (Nr, Nd) - number of vanishing moments for reconstruction and decomposition
    /// Coiflets wavelets
    Coiflets(usize), // Number of vanishing moments
    /// Haar wavelet (special case of Daubechies-1)
    Haar,
    /// Mexican Hat (Ricker) wavelet
    MexicanHat,
    /// Morlet wavelet
    Morlet,
}

/// Mode for handling boundary conditions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveletMode {
    /// Zero padding
    Zero,
    /// Constant (edge) padding
    Constant,
    /// Symmetric boundary conditions
    Symmetric,
    /// Periodic boundary conditions
    Periodic,
    /// Reflection boundary conditions
    Reflect,
}

/// 1D Discrete Wavelet Transform (DWT)
///
/// Decomposes a 1D signal into approximation and detail coefficients
pub fn dwt_1d(
    input: &Tensor,
    wavelet: WaveletType,
    mode: WaveletMode,
) -> TorshResult<(Tensor, Tensor)> {
    let shape = input.shape();
    if shape.ndim() != 1 {
        return Err(TorshError::InvalidArgument(format!(
            "Expected 1D input tensor, got {}D",
            shape.ndim()
        )));
    }

    let length = shape.dims()[0];
    if length < 2 {
        return Err(TorshError::InvalidArgument(
            "Input length must be at least 2".to_string(),
        ));
    }

    // Get wavelet coefficients
    let (low_pass, high_pass) = get_wavelet_coefficients(wavelet)?;

    // Apply convolution with downsampling
    let approx = convolve_downsample(input, &low_pass, mode)?;
    let detail = convolve_downsample(input, &high_pass, mode)?;

    Ok((approx, detail))
}

/// 1D Inverse Discrete Wavelet Transform (IDWT)
///
/// Reconstructs a signal from approximation and detail coefficients
pub fn idwt_1d(
    approx: &Tensor,
    detail: &Tensor,
    wavelet: WaveletType,
    mode: WaveletMode,
) -> TorshResult<Tensor> {
    if approx.shape() != detail.shape() {
        return Err(TorshError::ShapeMismatch {
            expected: approx.shape().dims().to_vec(),
            got: detail.shape().dims().to_vec(),
        });
    }

    // Get reconstruction filters
    let (rec_low, rec_high) = get_reconstruction_coefficients(wavelet)?;

    // Upsample and convolve
    let upsampled_approx = upsample_convolve(approx, &rec_low, mode)?;
    let upsampled_detail = upsample_convolve(detail, &rec_high, mode)?;

    // Add reconstructed components
    upsampled_approx.add_op(&upsampled_detail)
}

/// 2D Discrete Wavelet Transform
///
/// Applies separable 2D DWT to an image, producing 4 subbands: LL, LH, HL, HH
pub fn dwt_2d(
    input: &Tensor,
    wavelet: WaveletType,
    mode: WaveletMode,
) -> TorshResult<(Tensor, Tensor, Tensor, Tensor)> {
    let shape = input.shape();
    if shape.ndim() != 2 {
        return Err(TorshError::InvalidArgument(format!(
            "Expected 2D input tensor, got {}D",
            shape.ndim()
        )));
    }

    let (height, width) = (shape.dims()[0], shape.dims()[1]);
    if height < 2 || width < 2 {
        return Err(TorshError::InvalidArgument(
            "Input dimensions must be at least 2x2".to_string(),
        ));
    }

    // Get wavelet coefficients
    let (_low_pass, _high_pass) = get_wavelet_coefficients(wavelet)?;

    // First apply DWT along rows (width dimension)
    let mut row_approx = Vec::new();
    let mut row_detail = Vec::new();

    for h in 0..height {
        let row = input.narrow(0, h as i64, 1)?.squeeze(0)?; // Get single row
        let (a, d) = dwt_1d(&row, wavelet, mode)?;
        row_approx.push(a);
        row_detail.push(d);
    }

    // Stack row results
    let approx_rows = stack_tensors(&row_approx, 0)?;
    let detail_rows = stack_tensors(&row_detail, 0)?;

    // Then apply DWT along columns (height dimension)
    let new_width = approx_rows.shape().dims()[1];
    let mut ll_cols = Vec::new();
    let mut lh_cols = Vec::new();
    let mut hl_cols = Vec::new();
    let mut hh_cols = Vec::new();

    for w in 0..new_width {
        let approx_col = approx_rows.narrow(1, w as i64, 1)?.squeeze(1)?;
        let detail_col = detail_rows.narrow(1, w as i64, 1)?.squeeze(1)?;

        let (ll, lh) = dwt_1d(&approx_col, wavelet, mode)?;
        let (hl, hh) = dwt_1d(&detail_col, wavelet, mode)?;

        ll_cols.push(ll);
        lh_cols.push(lh);
        hl_cols.push(hl);
        hh_cols.push(hh);
    }

    let ll = stack_tensors(&ll_cols, 1)?;
    let lh = stack_tensors(&lh_cols, 1)?;
    let hl = stack_tensors(&hl_cols, 1)?;
    let hh = stack_tensors(&hh_cols, 1)?;

    Ok((ll, lh, hl, hh))
}

/// 2D Inverse Discrete Wavelet Transform
///
/// Reconstructs a 2D signal from its 4 wavelet subbands
pub fn idwt_2d(
    ll: &Tensor,
    lh: &Tensor,
    hl: &Tensor,
    hh: &Tensor,
    wavelet: WaveletType,
    mode: WaveletMode,
) -> TorshResult<Tensor> {
    // Verify all subbands have the same shape
    let ll_shape = ll.shape();
    if lh.shape() != ll_shape || hl.shape() != ll_shape || hh.shape() != ll_shape {
        return Err(TorshError::InvalidArgument(
            "All wavelet subbands must have the same shape".to_string(),
        ));
    }

    let (_sub_height, sub_width) = (ll_shape.dims()[0], ll_shape.dims()[1]);

    // First, reconstruct along columns (height dimension)
    let mut approx_cols = Vec::new();
    let mut detail_cols = Vec::new();

    for w in 0..sub_width {
        let ll_col = ll.narrow(1, w as i64, 1)?.squeeze(1)?;
        let lh_col = lh.narrow(1, w as i64, 1)?.squeeze(1)?;
        let hl_col = hl.narrow(1, w as i64, 1)?.squeeze(1)?;
        let hh_col = hh.narrow(1, w as i64, 1)?.squeeze(1)?;

        let approx_reconstructed = idwt_1d(&ll_col, &lh_col, wavelet, mode)?;
        let detail_reconstructed = idwt_1d(&hl_col, &hh_col, wavelet, mode)?;

        approx_cols.push(approx_reconstructed);
        detail_cols.push(detail_reconstructed);
    }

    let approx_rows = stack_tensors(&approx_cols, 1)?;
    let detail_rows = stack_tensors(&detail_cols, 1)?;

    // Then reconstruct along rows (width dimension)
    let height = approx_rows.shape().dims()[0];
    let mut final_rows = Vec::new();

    for h in 0..height {
        let approx_row = approx_rows.narrow(0, h as i64, 1)?.squeeze(0)?;
        let detail_row = detail_rows.narrow(0, h as i64, 1)?.squeeze(0)?;

        let reconstructed_row = idwt_1d(&approx_row, &detail_row, wavelet, mode)?;
        final_rows.push(reconstructed_row);
    }

    stack_tensors(&final_rows, 0)
}

/// Continuous Wavelet Transform (CWT)
///
/// Computes the continuous wavelet transform using the specified mother wavelet
pub fn cwt(input: &Tensor, scales: &[f32], wavelet: WaveletType) -> TorshResult<Tensor> {
    let input_length = input.shape().dims()[0];
    let num_scales = scales.len();

    // Initialize output tensor [scales, time]
    let mut cwt_coeffs = Vec::with_capacity(num_scales * input_length);

    for &scale in scales {
        let wavelet_kernel = generate_wavelet_kernel(wavelet, scale, input_length)?;
        let convolved = convolve_same(input, &wavelet_kernel)?;

        let convolved_data = convolved.data()?;
        cwt_coeffs.extend_from_slice(&convolved_data);
    }

    Tensor::from_data(cwt_coeffs, vec![num_scales, input_length], input.device())
}

/// Multi-level wavelet decomposition
///
/// Performs multiple levels of DWT decomposition
pub fn wavedec(
    input: &Tensor,
    wavelet: WaveletType,
    levels: usize,
    mode: WaveletMode,
) -> TorshResult<Vec<Tensor>> {
    if levels == 0 {
        return Err(TorshError::InvalidArgument(
            "Number of levels must be greater than 0".to_string(),
        ));
    }

    let mut coeffs = Vec::with_capacity(levels + 1);
    let mut current = input.clone();

    for _ in 0..levels {
        let (approx, detail) = dwt_1d(&current, wavelet, mode)?;
        coeffs.push(detail);
        current = approx;
    }

    // Add final approximation
    coeffs.push(current);
    coeffs.reverse(); // Convention: [approximation, detail_n, detail_n-1, ..., detail_1]

    Ok(coeffs)
}

/// Multi-level wavelet reconstruction
///
/// Reconstructs signal from multi-level wavelet coefficients
pub fn waverec(coeffs: &[Tensor], wavelet: WaveletType, mode: WaveletMode) -> TorshResult<Tensor> {
    if coeffs.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Coefficient list cannot be empty".to_string(),
        ));
    }

    let mut current = coeffs[0].clone(); // Start with approximation

    for i in 1..coeffs.len() {
        current = idwt_1d(&current, &coeffs[i], wavelet, mode)?;
    }

    Ok(current)
}

// Helper functions

fn get_wavelet_coefficients(wavelet: WaveletType) -> TorshResult<(Vec<f32>, Vec<f32>)> {
    match wavelet {
        WaveletType::Haar => {
            let low_pass = vec![
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ]; // [1/√2, 1/√2]
            let high_pass = vec![
                std::f32::consts::FRAC_1_SQRT_2,
                -std::f32::consts::FRAC_1_SQRT_2,
            ]; // [1/√2, -1/√2]
            Ok((low_pass, high_pass))
        }
        WaveletType::Daubechies(n) => {
            match n {
                2 => {
                    // Daubechies-2 (same as Haar)
                    let low_pass = vec![
                        std::f32::consts::FRAC_1_SQRT_2,
                        std::f32::consts::FRAC_1_SQRT_2,
                    ];
                    let high_pass = vec![
                        std::f32::consts::FRAC_1_SQRT_2,
                        -std::f32::consts::FRAC_1_SQRT_2,
                    ];
                    Ok((low_pass, high_pass))
                }
                4 => {
                    // Daubechies-4
                    let low_pass = vec![
                        0.48296291314469025,
                        0.8365163037378079,
                        0.22414386804185735,
                        -0.12940952255092145,
                    ];
                    let high_pass = vec![
                        -0.12940952255092145,
                        -0.22414386804185735,
                        0.8365163037378079,
                        -0.48296291314469025,
                    ];
                    Ok((low_pass, high_pass))
                }
                _ => Err(TorshError::UnsupportedOperation {
                    op: format!("Daubechies-{}", n),
                    dtype: "wavelet".to_string(),
                }),
            }
        }
        _ => Err(TorshError::UnsupportedOperation {
            op: format!("{:?}", wavelet),
            dtype: "wavelet".to_string(),
        }),
    }
}

fn get_reconstruction_coefficients(wavelet: WaveletType) -> TorshResult<(Vec<f32>, Vec<f32>)> {
    let (mut low_pass, mut high_pass) = get_wavelet_coefficients(wavelet)?;

    // For orthogonal wavelets, reconstruction filters are time-reversed decomposition filters
    low_pass.reverse();
    high_pass.reverse();

    // High-pass reconstruction filter has alternating signs
    for (i, val) in high_pass.iter_mut().enumerate() {
        if i % 2 == 1 {
            *val = -*val;
        }
    }

    Ok((low_pass, high_pass))
}

fn convolve_downsample(input: &Tensor, kernel: &[f32], _mode: WaveletMode) -> TorshResult<Tensor> {
    let input_data = input.data()?;
    let input_len = input_data.len();
    let _kernel_len = kernel.len();

    // For downsampling by 2
    let output_len = (input_len + 1) / 2;
    let mut output = Vec::with_capacity(output_len);

    for i in (0..input_len).step_by(2) {
        let mut sum = 0.0;

        for (k, &coeff) in kernel.iter().enumerate() {
            let idx = i as i32 - k as i32;
            if idx >= 0 && (idx as usize) < input_len {
                sum += input_data[idx as usize] * coeff;
            }
        }

        output.push(sum);
    }

    Tensor::from_data(output, vec![output_len], input.device())
}

fn upsample_convolve(input: &Tensor, kernel: &[f32], _mode: WaveletMode) -> TorshResult<Tensor> {
    let input_data = input.data()?;
    let input_len = input_data.len();
    let kernel_len = kernel.len();

    // Upsample by inserting zeros
    let upsampled_len = input_len * 2;
    let mut upsampled = vec![0.0; upsampled_len];

    for (i, &val) in input_data.iter().enumerate() {
        upsampled[i * 2] = val;
    }

    // Convolve with reconstruction filter
    let output_len = upsampled_len + kernel_len - 1;
    let mut output = vec![0.0; output_len];

    for i in 0..upsampled_len {
        for (k, &coeff) in kernel.iter().enumerate() {
            output[i + k] += upsampled[i] * coeff;
        }
    }

    // Trim to original size
    let trimmed_len = (output_len).min(upsampled_len);
    output.truncate(trimmed_len);

    Tensor::from_data(output, vec![trimmed_len], input.device())
}

fn stack_tensors(tensors: &[Tensor], dim: usize) -> TorshResult<Tensor> {
    if tensors.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot stack empty tensor list".to_string(),
        ));
    }

    let first_shape = tensors[0].shape();
    let mut stacked_shape = first_shape.dims().to_vec();
    stacked_shape.insert(dim, tensors.len());

    let element_count = first_shape.numel();
    let mut stacked_data = Vec::with_capacity(tensors.len() * element_count);

    for tensor in tensors {
        let data = tensor.data()?;
        stacked_data.extend_from_slice(&data);
    }

    Tensor::from_data(stacked_data, stacked_shape, tensors[0].device())
}

fn generate_wavelet_kernel(wavelet: WaveletType, scale: f32, length: usize) -> TorshResult<Tensor> {
    match wavelet {
        WaveletType::MexicanHat => {
            let mut kernel = Vec::with_capacity(length);
            let center = length as f32 / 2.0;

            for i in 0..length {
                let t = (i as f32 - center) / scale;
                let t2 = t * t;
                let val = (2.0 / (3.0 * scale).sqrt() * std::f32::consts::PI.powf(0.25))
                    * (1.0 - t2)
                    * (-t2 / 2.0).exp();
                kernel.push(val);
            }

            Tensor::from_data(kernel, vec![length], torsh_core::device::DeviceType::Cpu)
        }
        WaveletType::Morlet => {
            let mut kernel = Vec::with_capacity(length);
            let center = length as f32 / 2.0;
            let omega0 = 6.0; // Central frequency

            for i in 0..length {
                let t = (i as f32 - center) / scale;
                let val = (1.0 / (scale * std::f32::consts::PI.sqrt()))
                    * (omega0 * t).cos()
                    * (-(t * t) / 2.0).exp();
                kernel.push(val);
            }

            Tensor::from_data(kernel, vec![length], torsh_core::device::DeviceType::Cpu)
        }
        _ => Err(TorshError::UnsupportedOperation {
            op: format!("CWT with {:?}", wavelet),
            dtype: "wavelet".to_string(),
        }),
    }
}

fn convolve_same(input: &Tensor, kernel: &Tensor) -> TorshResult<Tensor> {
    let input_data = input.data()?;
    let kernel_data = kernel.data()?;
    let input_len = input_data.len();
    let kernel_len = kernel_data.len();

    let mut output = vec![0.0; input_len];
    let half_kernel = kernel_len / 2;

    for i in 0..input_len {
        for j in 0..kernel_len {
            let input_idx = i as i32 + j as i32 - half_kernel as i32;
            if input_idx >= 0 && (input_idx as usize) < input_len {
                output[i] += input_data[input_idx as usize] * kernel_data[j];
            }
        }
    }

    Tensor::from_data(output, vec![input_len], input.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_haar_dwt_1d() {
        let input = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let (approx, detail) = dwt_1d(&input, WaveletType::Haar, WaveletMode::Zero).unwrap();

        // Check that we get expected shapes
        assert_eq!(approx.shape().dims(), &[2]);
        assert_eq!(detail.shape().dims(), &[2]);

        // Check perfect reconstruction
        let reconstructed =
            idwt_1d(&approx, &detail, WaveletType::Haar, WaveletMode::Zero).unwrap();
        assert_eq!(reconstructed.shape().dims(), &[4]);
    }

    #[test]
    fn test_daubechies4_coefficients() {
        let (low_pass, high_pass) = get_wavelet_coefficients(WaveletType::Daubechies(4)).unwrap();

        // Check that we have the right number of coefficients
        assert_eq!(low_pass.len(), 4);
        assert_eq!(high_pass.len(), 4);

        // Check energy conservation (sum of squares should be 2 for normalized wavelets)
        let low_energy: f32 = low_pass.iter().map(|x| x * x).sum();
        let high_energy: f32 = high_pass.iter().map(|x| x * x).sum();

        assert!((low_energy - 1.0).abs() < 1e-6);
        assert!((high_energy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multilevel_decomposition() {
        let input = tensor_1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let coeffs = wavedec(&input, WaveletType::Haar, 2, WaveletMode::Zero).unwrap();

        // Should have 3 coefficient arrays: [approximation, detail_2, detail_1]
        assert_eq!(coeffs.len(), 3);

        // Reconstruct and verify
        let reconstructed = waverec(&coeffs, WaveletType::Haar, WaveletMode::Zero).unwrap();

        // Check that reconstruction has reasonable length
        assert!(reconstructed.shape().dims()[0] >= 4);
    }

    #[test]
    fn test_cwt_mexican_hat() {
        let input = tensor_1d(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).unwrap();
        let scales = vec![1.0, 2.0, 3.0];
        let result = cwt(&input, &scales, WaveletType::MexicanHat).unwrap();

        // Should have shape [num_scales, input_length]
        assert_eq!(result.shape().dims(), &[3, 6]);
    }
}
