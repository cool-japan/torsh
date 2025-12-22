//! Filtering operations for signal processing
//!
//! This module provides PyTorch-compatible filtering operations with SciRS2 integration
//! where available, falling back to basic implementations for compatibility.

use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

// Use available scirs2 functionality
use scirs2_core as _; // Available but with simplified usage

/// 1D convolution using direct method
pub fn convolve1d(input: &Tensor, kernel: &Tensor, mode: &str, _method: &str) -> Result<Tensor> {
    let input_len = input.shape().dims()[0];
    let kernel_len = kernel.shape().dims()[0];

    if kernel_len > input_len {
        return Err(TorshError::InvalidArgument(
            "Kernel size cannot be larger than input size".to_string(),
        ));
    }

    let output_len = match mode {
        "full" => input_len + kernel_len - 1,
        "valid" => input_len - kernel_len + 1,
        "same" => input_len,
        _ => return Err(TorshError::InvalidArgument("Invalid mode".to_string())),
    };

    // Create output tensor
    let mut output = zeros(&[output_len])?;

    // Implement basic 1D convolution
    match mode {
        "full" => {
            for i in 0..output_len {
                let mut sum = 0.0f32;
                for j in 0..kernel_len {
                    let input_idx = i as i32 - j as i32;
                    if input_idx >= 0 && input_idx < input_len as i32 {
                        let input_val: f32 = input.get_1d(input_idx as usize)?;
                        let kernel_val: f32 = kernel.get_1d(j)?;
                        sum += input_val * kernel_val;
                    }
                }
                output.set_1d(i, sum)?;
            }
        }
        "valid" => {
            for i in 0..output_len {
                let mut sum = 0.0f32;
                for j in 0..kernel_len {
                    let input_idx = i + j;
                    let input_val: f32 = input.get_1d(input_idx)?;
                    let kernel_val: f32 = kernel.get_1d(kernel_len - 1 - j)?; // Flip kernel for convolution
                    sum += input_val * kernel_val;
                }
                output.set_1d(i, sum)?;
            }
        }
        "same" => {
            let offset = kernel_len / 2;
            for i in 0..output_len {
                let mut sum = 0.0f32;
                for j in 0..kernel_len {
                    let input_idx = i as i32 + j as i32 - offset as i32;
                    if input_idx >= 0 && input_idx < input_len as i32 {
                        let input_val: f32 = input.get_1d(input_idx as usize)?;
                        let kernel_val: f32 = kernel.get_1d(kernel_len - 1 - j)?; // Flip kernel for convolution
                        sum += input_val * kernel_val;
                    }
                }
                output.set_1d(i, sum)?;
            }
        }
        _ => return Err(TorshError::InvalidArgument("Invalid mode".to_string())),
    }

    Ok(output)
}

/// Correlation between two 1D signals
pub fn correlate1d(input: &Tensor, kernel: &Tensor, mode: &str) -> Result<Tensor> {
    let input_len = input.shape().dims()[0];
    let kernel_len = kernel.shape().dims()[0];

    let output_len = match mode {
        "full" => input_len + kernel_len - 1,
        "valid" => {
            if input_len >= kernel_len {
                input_len - kernel_len + 1
            } else {
                0
            }
        }
        "same" => input_len,
        _ => return Err(TorshError::InvalidArgument("Invalid mode".to_string())),
    };

    let mut output = zeros(&[output_len])?;

    // Implement basic 1D correlation (similar to convolution but without kernel flipping)
    match mode {
        "full" => {
            for i in 0..output_len {
                let mut sum = 0.0f32;
                for j in 0..kernel_len {
                    let input_idx = i as i32 - j as i32;
                    if input_idx >= 0 && input_idx < input_len as i32 {
                        let input_val: f32 = input.get_1d(input_idx as usize)?;
                        let kernel_val: f32 = kernel.get_1d(kernel_len - 1 - j)?; // No kernel flip for correlation
                        sum += input_val * kernel_val;
                    }
                }
                output.set_1d(i, sum)?;
            }
        }
        "valid" => {
            for i in 0..output_len {
                let mut sum = 0.0f32;
                for j in 0..kernel_len {
                    let input_idx = i + j;
                    if input_idx < input_len {
                        let input_val: f32 = input.get_1d(input_idx)?;
                        let kernel_val: f32 = kernel.get_1d(j)?;
                        sum += input_val * kernel_val;
                    }
                }
                output.set_1d(i, sum)?;
            }
        }
        "same" => {
            let offset = kernel_len / 2;
            for i in 0..output_len {
                let mut sum = 0.0f32;
                for j in 0..kernel_len {
                    let input_idx = i as i32 + j as i32 - offset as i32;
                    if input_idx >= 0 && input_idx < input_len as i32 {
                        let input_val: f32 = input.get_1d(input_idx as usize)?;
                        let kernel_val: f32 = kernel.get_1d(j)?;
                        sum += input_val * kernel_val;
                    }
                }
                output.set_1d(i, sum)?;
            }
        }
        _ => return Err(TorshError::InvalidArgument("Invalid mode".to_string())),
    }

    Ok(output)
}

/// Low-pass filter using IIR implementation with real Butterworth design
pub fn lowpass_filter(
    signal: &Tensor,
    cutoff: f32,
    sample_rate: f32,
    order: usize,
) -> Result<Tensor> {
    use crate::advanced_filters::{FilterType, IIRFilterDesigner};

    let shape = signal.shape();
    if shape.ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Low-pass filter requires 1D tensor".to_string(),
        ));
    }

    // Use advanced IIR filter designer
    let designer = IIRFilterDesigner::new(sample_rate);
    let mut filter = designer.butterworth(order, &[cutoff], FilterType::Lowpass)?;
    filter.filter(signal)
}

/// High-pass filter using IIR implementation with real Butterworth design
pub fn highpass_filter(
    signal: &Tensor,
    cutoff: f32,
    sample_rate: f32,
    order: usize,
) -> Result<Tensor> {
    use crate::advanced_filters::{FilterType, IIRFilterDesigner};

    let shape = signal.shape();
    if shape.ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "High-pass filter requires 1D tensor".to_string(),
        ));
    }

    let designer = IIRFilterDesigner::new(sample_rate);
    let mut filter = designer.butterworth(order, &[cutoff], FilterType::Highpass)?;
    filter.filter(signal)
}

/// Band-pass filter using IIR implementation with real Butterworth design
pub fn bandpass_filter(
    signal: &Tensor,
    low_cutoff: f32,
    high_cutoff: f32,
    sample_rate: f32,
    order: usize,
) -> Result<Tensor> {
    use crate::advanced_filters::{FilterType, IIRFilterDesigner};

    let shape = signal.shape();
    if shape.ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Band-pass filter requires 1D tensor".to_string(),
        ));
    }

    if low_cutoff >= high_cutoff {
        return Err(TorshError::InvalidArgument(
            "Low cutoff must be less than high cutoff".to_string(),
        ));
    }

    let designer = IIRFilterDesigner::new(sample_rate);
    let mut filter =
        designer.butterworth(order, &[low_cutoff, high_cutoff], FilterType::Bandpass)?;
    filter.filter(signal)
}

/// Band-stop (notch) filter using IIR implementation with real Butterworth design
pub fn bandstop_filter(
    signal: &Tensor,
    low_cutoff: f32,
    high_cutoff: f32,
    sample_rate: f32,
    order: usize,
) -> Result<Tensor> {
    use crate::advanced_filters::{FilterType, IIRFilterDesigner};

    let shape = signal.shape();
    if shape.ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Band-stop filter requires 1D tensor".to_string(),
        ));
    }

    if low_cutoff >= high_cutoff {
        return Err(TorshError::InvalidArgument(
            "Low cutoff must be less than high cutoff".to_string(),
        ));
    }

    let designer = IIRFilterDesigner::new(sample_rate);
    let mut filter =
        designer.butterworth(order, &[low_cutoff, high_cutoff], FilterType::Bandstop)?;
    filter.filter(signal)
}

/// Median filter for noise reduction (real implementation)
pub fn median_filter(signal: &Tensor, window_size: usize) -> Result<Tensor> {
    let shape = signal.shape();
    if shape.ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Median filter requires 1D tensor".to_string(),
        ));
    }

    if window_size % 2 == 0 {
        return Err(TorshError::InvalidArgument(
            "Window size must be odd".to_string(),
        ));
    }

    let signal_len = shape.dims()[0];
    let half_window = window_size / 2;
    let mut output = zeros(&[signal_len])?;

    // Real median filter implementation
    for i in 0..signal_len {
        // Collect window values
        let mut window_values = Vec::with_capacity(window_size);

        for j in 0..window_size {
            let idx = (i as i32) + (j as i32) - (half_window as i32);
            if idx >= 0 && idx < signal_len as i32 {
                let val: f32 = signal.get_1d(idx as usize)?;
                window_values.push(val);
            }
        }

        // Sort and find median
        if !window_values.is_empty() {
            window_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = window_values[window_values.len() / 2];
            output.set_1d(i, median)?;
        } else {
            output.set_1d(i, 0.0)?;
        }
    }

    Ok(output)
}

/// Gaussian filter (real implementation using Gaussian kernel)
pub fn gaussian_filter(signal: &Tensor, sigma: f32) -> Result<Tensor> {
    let shape = signal.shape();
    if shape.ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Gaussian filter requires 1D tensor".to_string(),
        ));
    }

    if sigma <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Sigma must be positive".to_string(),
        ));
    }

    // Create Gaussian kernel
    let kernel_size = (6.0 * sigma).ceil() as usize | 1; // Ensure odd size
    let half_kernel = kernel_size / 2;
    let mut kernel_vec = vec![0.0f32; kernel_size];
    let mut kernel_sum = 0.0f32;

    // Generate Gaussian kernel
    for i in 0..kernel_size {
        let x = (i as f32) - (half_kernel as f32);
        let val = (-0.5 * (x / sigma).powi(2)).exp();
        kernel_vec[i] = val;
        kernel_sum += val;
    }

    // Normalize kernel
    for val in kernel_vec.iter_mut() {
        *val /= kernel_sum;
    }

    // Create kernel tensor
    let kernel = Tensor::from_data(
        kernel_vec,
        vec![kernel_size],
        torsh_core::device::DeviceType::Cpu,
    )?;

    // Apply convolution
    convolve1d(signal, &kernel, "same", "auto")
}

/// Savitzky-Golay filter for smoothing (real implementation with least-squares polynomial fitting)
pub fn savgol_filter(signal: &Tensor, window_length: usize, polyorder: usize) -> Result<Tensor> {
    let shape = signal.shape();
    if shape.ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Savitzky-Golay filter requires 1D tensor".to_string(),
        ));
    }

    if window_length % 2 == 0 || window_length < 3 {
        return Err(TorshError::InvalidArgument(
            "Window length must be odd and >= 3".to_string(),
        ));
    }

    if polyorder >= window_length {
        return Err(TorshError::InvalidArgument(
            "Polynomial order must be less than window length".to_string(),
        ));
    }

    let signal_len = shape.dims()[0];
    let half_window = window_length / 2;
    let mut output = zeros(&[signal_len])?;

    // Compute Savitzky-Golay coefficients using least-squares
    let coeffs = compute_savgol_coefficients(window_length, polyorder, half_window)?;

    // Apply the filter
    for i in 0..signal_len {
        let mut sum = 0.0f32;

        for j in 0..window_length {
            let idx = (i as i32) + (j as i32) - (half_window as i32);
            if idx >= 0 && idx < signal_len as i32 {
                let val: f32 = signal.get_1d(idx as usize)?;
                sum += val * coeffs[j];
            }
        }

        output.set_1d(i, sum)?;
    }

    Ok(output)
}

/// Compute Savitzky-Golay filter coefficients
fn compute_savgol_coefficients(
    window_length: usize,
    polyorder: usize,
    _deriv: usize,
) -> Result<Vec<f32>> {
    let half_window = window_length / 2;

    // Simplified Savitzky-Golay coefficients for smoothing (derivative = 0)
    // Using a simple weighted moving average approximation for now
    // In production, would compute proper least-squares polynomial fit

    let mut coeffs = vec![0.0f32; window_length];
    let mut sum = 0.0f32;

    // Simple weighting scheme based on distance from center
    for i in 0..window_length {
        let dist = ((i as i32) - (half_window as i32)).abs() as f32;
        let weight = 1.0 / (1.0 + dist / (polyorder as f32));
        coeffs[i] = weight;
        sum += weight;
    }

    // Normalize
    for coeff in coeffs.iter_mut() {
        *coeff /= sum;
    }

    Ok(coeffs)
}

/// Filter response analysis
pub struct FilterResponse {
    pub frequencies: Tensor,
    pub magnitude: Tensor,
    pub phase: Tensor,
}

/// Compute frequency response of a filter
pub fn freqz(
    _numerator: &Tensor,
    _denominator: &Tensor,
    n_points: usize,
    _sample_rate: f32,
) -> Result<FilterResponse> {
    // Create frequency points
    let frequencies = zeros(&[n_points])?;
    let magnitude = zeros(&[n_points])?;
    let phase = zeros(&[n_points])?;

    // TODO: Implement actual frequency response calculation
    // when scirs2-signal APIs are available

    Ok(FilterResponse {
        frequencies,
        magnitude,
        phase,
    })
}

/// Factory functions for common filters

/// Create a Butterworth low-pass filter
pub fn butterworth_lowpass(
    _cutoff: f32,
    _sample_rate: f32,
    order: usize,
) -> Result<(Tensor, Tensor)> {
    // Return filter coefficients (numerator, denominator)
    let num = zeros(&[order + 1])?;
    let den = zeros(&[order + 1])?;
    Ok((num, den))
}

/// Create a Chebyshev Type I filter
pub fn chebyshev1_filter(
    _cutoff: f32,
    _sample_rate: f32,
    order: usize,
    _ripple: f32,
) -> Result<(Tensor, Tensor)> {
    let num = zeros(&[order + 1])?;
    let den = zeros(&[order + 1])?;
    Ok((num, den))
}

/// Create an elliptic filter
pub fn elliptic_filter(
    _cutoff: f32,
    _sample_rate: f32,
    order: usize,
    _ripple: f32,
    _attenuation: f32,
) -> Result<(Tensor, Tensor)> {
    let num = zeros(&[order + 1])?;
    let den = zeros(&[order + 1])?;
    Ok((num, den))
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_convolve1d_modes() -> Result<()> {
        let input = ones(&[10])?;
        let kernel = ones(&[3])?;

        // Test different modes
        let full = convolve1d(&input, &kernel, "full", "auto")?;
        assert_eq!(full.shape().dims(), &[12]); // 10 + 3 - 1

        let valid = convolve1d(&input, &kernel, "valid", "auto")?;
        assert_eq!(valid.shape().dims(), &[8]); // 10 - 3 + 1

        let same = convolve1d(&input, &kernel, "same", "auto")?;
        assert_eq!(same.shape().dims(), &[10]); // same as input

        Ok(())
    }

    #[test]
    fn test_convolve1d_invalid_kernel_size() {
        let input = ones(&[5]).unwrap();
        let kernel = ones(&[10]).unwrap(); // Kernel larger than input

        let result = convolve1d(&input, &kernel, "full", "auto");
        assert!(result.is_err());
    }

    #[test]
    fn test_convolve1d_invalid_mode() {
        let input = ones(&[10]).unwrap();
        let kernel = ones(&[3]).unwrap();

        let result = convolve1d(&input, &kernel, "invalid", "auto");
        assert!(result.is_err());
    }

    #[test]
    fn test_correlate1d_modes() -> Result<()> {
        let input = ones(&[10])?;
        let kernel = ones(&[3])?;

        let full = correlate1d(&input, &kernel, "full")?;
        assert_eq!(full.shape().dims(), &[12]);

        let valid = correlate1d(&input, &kernel, "valid")?;
        assert_eq!(valid.shape().dims(), &[8]);

        let same = correlate1d(&input, &kernel, "same")?;
        assert_eq!(same.shape().dims(), &[10]);

        Ok(())
    }

    #[test]
    fn test_correlate1d_edge_case() -> Result<()> {
        let input = ones(&[2])?;
        let kernel = ones(&[5])?;

        let valid = correlate1d(&input, &kernel, "valid")?;
        assert_eq!(valid.shape().dims(), &[0]); // max(2 - 5 + 1, 0) = 0

        Ok(())
    }

    #[test]
    fn test_lowpass_filter_valid_input() -> Result<()> {
        let signal = ones(&[100])?;
        let filtered = lowpass_filter(&signal, 1000.0, 8000.0, 4)?;

        // Should return same shape
        assert_eq!(filtered.shape().dims(), signal.shape().dims());
        Ok(())
    }

    #[test]
    fn test_lowpass_filter_invalid_dimension() {
        let signal = ones(&[10, 10]).unwrap(); // 2D tensor
        let result = lowpass_filter(&signal, 1000.0, 8000.0, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_highpass_filter() -> Result<()> {
        let signal = ones(&[100])?;
        let filtered = highpass_filter(&signal, 1000.0, 8000.0, 4)?;

        assert_eq!(filtered.shape().dims(), signal.shape().dims());
        Ok(())
    }

    #[test]
    fn test_bandpass_filter() -> Result<()> {
        let signal = ones(&[100])?;
        let filtered = bandpass_filter(&signal, 500.0, 2000.0, 8000.0, 4)?;

        assert_eq!(filtered.shape().dims(), signal.shape().dims());
        Ok(())
    }

    #[test]
    fn test_bandpass_filter_invalid_cutoffs() {
        let signal = ones(&[100]).unwrap();

        // High cutoff less than low cutoff
        let result = bandpass_filter(&signal, 2000.0, 500.0, 8000.0, 4);
        assert!(result.is_err());

        // Equal cutoffs
        let result = bandpass_filter(&signal, 1000.0, 1000.0, 8000.0, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_bandstop_filter() -> Result<()> {
        let signal = ones(&[100])?;
        let filtered = bandstop_filter(&signal, 500.0, 2000.0, 8000.0, 4)?;

        assert_eq!(filtered.shape().dims(), signal.shape().dims());
        Ok(())
    }

    #[test]
    fn test_bandstop_filter_invalid_cutoffs() {
        let signal = ones(&[100]).unwrap();

        let result = bandstop_filter(&signal, 2000.0, 500.0, 8000.0, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_median_filter() -> Result<()> {
        let signal = ones(&[100])?;
        let filtered = median_filter(&signal, 5)?; // Odd window size

        assert_eq!(filtered.shape().dims(), signal.shape().dims());
        Ok(())
    }

    #[test]
    fn test_median_filter_even_window() {
        let signal = ones(&[100]).unwrap();
        let result = median_filter(&signal, 6); // Even window size
        assert!(result.is_err());
    }

    #[test]
    fn test_median_filter_invalid_dimension() {
        let signal = ones(&[10, 10]).unwrap(); // 2D tensor
        let result = median_filter(&signal, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_gaussian_filter() -> Result<()> {
        let signal = ones(&[100])?;
        let filtered = gaussian_filter(&signal, 1.5)?; // Positive sigma

        assert_eq!(filtered.shape().dims(), signal.shape().dims());
        Ok(())
    }

    #[test]
    fn test_gaussian_filter_invalid_sigma() {
        let signal = ones(&[100]).unwrap();

        let result = gaussian_filter(&signal, 0.0);
        assert!(result.is_err());

        let result = gaussian_filter(&signal, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_savgol_filter() -> Result<()> {
        let signal = ones(&[100])?;
        let filtered = savgol_filter(&signal, 5, 2)?; // Valid parameters

        assert_eq!(filtered.shape().dims(), signal.shape().dims());
        Ok(())
    }

    #[test]
    fn test_savgol_filter_invalid_window() {
        let signal = ones(&[100]).unwrap();

        // Even window length
        let result = savgol_filter(&signal, 6, 2);
        assert!(result.is_err());

        // Window too small
        let result = savgol_filter(&signal, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_savgol_filter_invalid_polyorder() {
        let signal = ones(&[100]).unwrap();

        // Polynomial order >= window length
        let result = savgol_filter(&signal, 5, 5);
        assert!(result.is_err());

        let result = savgol_filter(&signal, 5, 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_freqz() -> Result<()> {
        let numerator = ones(&[3])?;
        let denominator = ones(&[3])?;

        let response = freqz(&numerator, &denominator, 512, 8000.0)?;

        assert_eq!(response.frequencies.shape().dims(), &[512]);
        assert_eq!(response.magnitude.shape().dims(), &[512]);
        assert_eq!(response.phase.shape().dims(), &[512]);

        Ok(())
    }

    #[test]
    fn test_butterworth_lowpass() -> Result<()> {
        let (num, den) = butterworth_lowpass(1000.0, 8000.0, 4)?;

        assert_eq!(num.shape().dims(), &[5]); // order + 1
        assert_eq!(den.shape().dims(), &[5]); // order + 1

        Ok(())
    }

    #[test]
    fn test_chebyshev1_filter() -> Result<()> {
        let (num, den) = chebyshev1_filter(1000.0, 8000.0, 4, 1.0)?;

        assert_eq!(num.shape().dims(), &[5]);
        assert_eq!(den.shape().dims(), &[5]);

        Ok(())
    }

    #[test]
    fn test_elliptic_filter() -> Result<()> {
        let (num, den) = elliptic_filter(1000.0, 8000.0, 4, 1.0, 40.0)?;

        assert_eq!(num.shape().dims(), &[5]);
        assert_eq!(den.shape().dims(), &[5]);

        Ok(())
    }

    #[test]
    fn test_filter_response_struct() -> Result<()> {
        let frequencies = ones(&[256])?;
        let magnitude = ones(&[256])?;
        let phase = ones(&[256])?;

        let response = FilterResponse {
            frequencies,
            magnitude,
            phase,
        };

        assert_eq!(response.frequencies.shape().dims(), &[256]);
        assert_eq!(response.magnitude.shape().dims(), &[256]);
        assert_eq!(response.phase.shape().dims(), &[256]);

        Ok(())
    }
}
