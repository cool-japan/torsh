//! Window functions for signal processing
//!
//! This module provides PyTorch-compatible window functions built on top of
//! scirs2-signal for high-performance signal processing operations.

use scirs2_core::constants::math::PI;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{
    creation::{ones, zeros},
    Tensor,
};

// Use available scirs2 functionality
use scirs2_core as _; // Available but with simplified usage

/// Window type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Window {
    Rectangular,
    Hamming,
    Hann,
    Blackman,
    BlackmanHarris,
    Bartlett,
    Kaiser(f32),
    Gaussian(f32),
    Tukey(f32),
    Cosine,
    Exponential(f32),
}

/// Create a window function of the given type and length
pub fn window(window_type: Window, n: usize, periodic: bool) -> Result<Tensor> {
    match window_type {
        Window::Rectangular => rectangular_window(n),
        Window::Hamming => hamming_window(n, periodic),
        Window::Hann => hann_window(n, periodic),
        Window::Blackman => blackman_window(n, periodic),
        Window::BlackmanHarris => blackman_harris_window(n, periodic),
        Window::Bartlett => bartlett_window(n, periodic),
        Window::Kaiser(beta) => kaiser_window(n, beta, periodic),
        Window::Gaussian(std) => gaussian_window(n, std, periodic),
        Window::Tukey(alpha) => tukey_window(n, alpha, periodic),
        Window::Cosine => cosine_window(n, periodic),
        Window::Exponential(tau) => exponential_window(n, tau),
    }
}

/// Rectangular (boxcar) window
pub fn rectangular_window(n: usize) -> Result<Tensor> {
    Ok(ones(&[n])?)
}

/// Hamming window
pub fn hamming_window(n: usize, periodic: bool) -> Result<Tensor> {
    if n == 0 {
        return Ok(zeros(&[0])?);
    }
    if n == 1 {
        return Ok(ones(&[1])?);
    }

    let n_adj = if periodic { n } else { n - 1 };
    let mut window = zeros(&[n])?;

    for i in 0..n {
        let value = 0.54 - 0.46 * ((2.0 * PI * i as f64) / n_adj as f64).cos();
        window.set_1d(i, value as f32)?;
    }

    Ok(window)
}

/// Hann (Hanning) window
pub fn hann_window(n: usize, periodic: bool) -> Result<Tensor> {
    if n == 0 {
        return Ok(zeros(&[0])?);
    }
    if n == 1 {
        return Ok(ones(&[1])?);
    }

    let n_adj = if periodic { n } else { n - 1 };
    let mut window = zeros(&[n])?;

    for i in 0..n {
        let value = 0.5 - 0.5 * ((2.0 * PI * i as f64) / n_adj as f64).cos();
        window.set_1d(i, value as f32)?;
    }

    Ok(window)
}

/// Blackman window
pub fn blackman_window(n: usize, periodic: bool) -> Result<Tensor> {
    if n == 0 {
        return Ok(zeros(&[0])?);
    }
    if n == 1 {
        return Ok(ones(&[1])?);
    }

    let n_adj = if periodic { n } else { n - 1 };
    let mut window = zeros(&[n])?;

    let a0 = 0.42;
    let a1 = 0.5;
    let a2 = 0.08;

    for i in 0..n {
        let value = a0 - a1 * ((2.0 * PI * i as f64) / n_adj as f64).cos()
            + a2 * ((4.0 * PI * i as f64) / n_adj as f64).cos();
        window.set_1d(i, value as f32)?;
    }

    Ok(window)
}

/// Blackman-Harris window
pub fn blackman_harris_window(n: usize, periodic: bool) -> Result<Tensor> {
    if n == 0 {
        return Ok(zeros(&[0])?);
    }
    if n == 1 {
        return Ok(ones(&[1])?);
    }

    let n_adj = if periodic { n } else { n - 1 };
    let mut window = zeros(&[n])?;

    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;

    for i in 0..n {
        let value = a0 - a1 * ((2.0 * PI * i as f64) / n_adj as f64).cos()
            + a2 * ((4.0 * PI * i as f64) / n_adj as f64).cos()
            - a3 * ((6.0 * PI * i as f64) / n_adj as f64).cos();
        window.set_1d(i, value as f32)?;
    }

    Ok(window)
}

/// Bartlett (triangular) window
pub fn bartlett_window(n: usize, periodic: bool) -> Result<Tensor> {
    if n == 0 {
        return Ok(zeros(&[0])?);
    }
    if n == 1 {
        return Ok(ones(&[1])?);
    }

    let n_adj = if periodic { n + 1 } else { n };
    let mut window = zeros(&[n])?;

    for i in 0..n {
        let value = if i < n_adj / 2 {
            2.0 * i as f64 / (n_adj - 1) as f64
        } else {
            2.0 - 2.0 * i as f64 / (n_adj - 1) as f64
        };
        window.set_1d(i, value as f32)?;
    }

    Ok(window)
}

/// Kaiser window with beta parameter
pub fn kaiser_window(n: usize, beta: f32, periodic: bool) -> Result<Tensor> {
    if n == 0 {
        return Ok(zeros(&[0])?);
    }
    if n == 1 {
        return Ok(ones(&[1])?);
    }

    let n_adj = if periodic { n } else { n - 1 };
    let mut window = zeros(&[n])?;

    let alpha = (n_adj as f64) / 2.0;
    let beta_f64 = beta as f64;

    for i in 0..n {
        let arg = beta_f64 * ((1.0 - ((i as f64 - alpha) / alpha).powi(2)).sqrt());
        let value = modified_bessel_i0(arg) / modified_bessel_i0(beta_f64);
        window.set_1d(i, value as f32)?;
    }

    Ok(window)
}

/// Gaussian window with standard deviation
pub fn gaussian_window(n: usize, std: f32, periodic: bool) -> Result<Tensor> {
    if n == 0 {
        return Ok(zeros(&[0])?);
    }
    if n == 1 {
        return Ok(ones(&[1])?);
    }

    let n_adj = if periodic { n } else { n - 1 };
    let mut window = zeros(&[n])?;

    let sigma = std as f64 * n_adj as f64 / 2.0;
    let center = (n_adj as f64) / 2.0;

    for i in 0..n {
        let x = i as f64 - center;
        let value = (-(x * x) / (2.0 * sigma * sigma)).exp();
        window.set_1d(i, value as f32)?;
    }

    Ok(window)
}

/// Tukey (tapered cosine) window with alpha parameter
pub fn tukey_window(n: usize, alpha: f32, periodic: bool) -> Result<Tensor> {
    if n == 0 {
        return Ok(zeros(&[0])?);
    }
    if n == 1 {
        return Ok(ones(&[1])?);
    }

    if alpha <= 0.0 {
        return rectangular_window(n);
    }
    if alpha >= 1.0 {
        return hann_window(n, periodic);
    }

    let n_adj = if periodic { n } else { n - 1 };
    let mut window = zeros(&[n])?;

    let width = (alpha as f64 * n_adj as f64 / 2.0) as usize;

    for i in 0..n {
        let value = if i < width {
            0.5 * (1.0 - ((PI * i as f64) / width as f64).cos())
        } else if i < n_adj - width {
            1.0
        } else {
            0.5 * (1.0 - ((PI * (n_adj - i) as f64) / width as f64).cos())
        };
        window.set_1d(i, value as f32)?;
    }

    Ok(window)
}

/// Cosine window
pub fn cosine_window(n: usize, periodic: bool) -> Result<Tensor> {
    if n == 0 {
        return Ok(zeros(&[0])?);
    }
    if n == 1 {
        return Ok(ones(&[1])?);
    }

    let n_adj = if periodic { n } else { n - 1 };
    let mut window = zeros(&[n])?;

    for i in 0..n {
        let value = ((PI * i as f64) / n_adj as f64).sin();
        window.set_1d(i, value as f32)?;
    }

    Ok(window)
}

/// Exponential window with decay parameter tau
pub fn exponential_window(n: usize, tau: f32) -> Result<Tensor> {
    if n == 0 {
        return Ok(zeros(&[0])?);
    }

    let mut window = zeros(&[n])?;
    let center = (n as f64 - 1.0) / 2.0;
    let tau_f64 = tau as f64;

    for i in 0..n {
        let value = (-(i as f64 - center).abs() / tau_f64).exp();
        window.set_1d(i, value as f32)?;
    }

    Ok(window)
}

/// Modified Bessel function of the first kind, order 0
fn modified_bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half_sq = (x / 2.0).powi(2);

    for k in 1..50 {
        term *= x_half_sq / (k as f64).powi(2);
        sum += term;
        if term < 1e-10 * sum {
            break;
        }
    }

    sum
}

/// Normalize a window by its sum
pub fn normalize_window(window: &Tensor, norm_type: &str) -> Result<Tensor> {
    let window_len = window.shape().dims()[0];
    let mut normalized = zeros(&[window_len])?;

    match norm_type {
        "magnitude" => {
            let sum = window.sum()?;
            let sum_val = sum.item().unwrap();

            // Manually divide each element to avoid tensor shape issues
            for i in 0..window_len {
                let val = window.get_1d(i)?;
                normalized.set_1d(i, val / sum_val)?;
            }
            Ok(normalized)
        }
        "power" => {
            let power_sum = window.pow_scalar(2.0)?.sum()?;
            let norm_factor = power_sum.sqrt()?;
            let norm_val = norm_factor.item().unwrap();

            // Manually divide each element to avoid tensor shape issues
            for i in 0..window_len {
                let val = window.get_1d(i)?;
                normalized.set_1d(i, val / norm_val)?;
            }
            Ok(normalized)
        }
        _ => Err(TorshError::InvalidArgument(format!(
            "Unknown normalization type: {}",
            norm_type
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hamming_window() {
        let window = hamming_window(10, false).unwrap();
        assert_eq!(window.shape().dims(), &[10]);

        // Check symmetry
        for i in 0..5 {
            assert_relative_eq!(
                window.get_1d(i).unwrap(),
                window.get_1d(9 - i).unwrap(),
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn test_hann_window() {
        let window = hann_window(10, false).unwrap();
        assert_eq!(window.shape().dims(), &[10]);

        // Check endpoints
        assert_relative_eq!(window.get_1d(0).unwrap(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(window.get_1d(9).unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_window_normalization() -> Result<()> {
        let window = hamming_window(10, false)?;
        let normalized = normalize_window(&window, "magnitude")?;
        let sum = normalized.sum()?;
        assert_relative_eq!(sum.item().unwrap(), 1.0, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_periodic_vs_symmetric() -> Result<()> {
        let n = 10;
        let periodic = hamming_window(n, true)?;
        let symmetric = hamming_window(n, false)?;

        // Periodic and symmetric should be different
        let diff = periodic.sub(&symmetric)?.abs()?.sum()?;
        assert!(diff.item().unwrap() > 0.01);
        Ok(())
    }

    #[test]
    fn test_blackman_window() -> Result<()> {
        let window = blackman_window(8, false)?;
        assert_eq!(window.shape().dims(), &[8]);

        // Check symmetry
        for i in 0..4 {
            assert_relative_eq!(window.get_1d(i)?, window.get_1d(7 - i)?, epsilon = 1e-6);
        }

        // Check that endpoints are near zero (but not exactly zero)
        assert!(window.get_1d(0)? < 0.1);
        assert!(window.get_1d(7)? < 0.1);
        Ok(())
    }

    #[test]
    fn test_bartlett_window() -> Result<()> {
        let window = bartlett_window(9, false)?;
        assert_eq!(window.shape().dims(), &[9]);

        // Check endpoints for Bartlett
        assert_relative_eq!(window.get_1d(0)?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(window.get_1d(8)?, 0.0, epsilon = 1e-6);

        // Check center value
        assert_relative_eq!(window.get_1d(4)?, 1.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_kaiser_window() -> Result<()> {
        let beta = 5.0;
        let window = kaiser_window(10, beta, false)?;
        assert_eq!(window.shape().dims(), &[10]);

        // Kaiser window should have maximum at center
        let center_value = window.get_1d(4)?; // Center for n=10, symmetric
        for i in 0..10 {
            assert!(window.get_1d(i)? <= center_value + 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_gaussian_window() -> Result<()> {
        let std = 0.4;
        let window = gaussian_window(10, std, false)?;
        assert_eq!(window.shape().dims(), &[10]);

        // Gaussian should be symmetric
        for i in 0..5 {
            assert_relative_eq!(window.get_1d(i)?, window.get_1d(9 - i)?, epsilon = 1e-6);
        }

        // Maximum should be near the center
        let center_value = (window.get_1d(4)? + window.get_1d(5)?) / 2.0;
        for i in 0..10 {
            assert!(window.get_1d(i)? <= center_value + 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_tukey_window() -> Result<()> {
        let alpha = 0.5;
        let window = tukey_window(10, alpha, false)?;
        assert_eq!(window.shape().dims(), &[10]);

        // Tukey window should have a flat top region
        let center_values = vec![window.get_1d(4)?, window.get_1d(5)?];
        for value in center_values {
            assert!(value > 0.9); // Should be close to 1.0 in the flat region
        }
        Ok(())
    }

    #[test]
    fn test_cosine_window() -> Result<()> {
        let window = cosine_window(10, false)?;
        assert_eq!(window.shape().dims(), &[10]);

        // Check symmetry
        for i in 0..5 {
            assert_relative_eq!(window.get_1d(i)?, window.get_1d(9 - i)?, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_exponential_window() -> Result<()> {
        let tau = 2.0;
        let window = exponential_window(10, tau)?;
        assert_eq!(window.shape().dims(), &[10]);

        // Maximum should be at center
        let center_idx = 4; // For n=10, center is at index 4.5, so check indices 4 and 5
        let center_value = window.get_1d(center_idx)?;
        for i in 0..10 {
            assert!(window.get_1d(i)? <= center_value + 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_window_enum() -> Result<()> {
        let n = 8;

        // Test different window types through the enum interface
        let windows = vec![
            Window::Rectangular,
            Window::Hamming,
            Window::Hann,
            Window::Blackman,
            Window::BlackmanHarris,
            Window::Bartlett,
            Window::Kaiser(5.0),
            Window::Gaussian(0.4),
            Window::Tukey(0.5),
            Window::Cosine,
            Window::Exponential(2.0),
        ];

        for window_type in windows {
            let w = window(window_type, n, false)?;
            assert_eq!(w.shape().dims(), &[n]);

            // All windows should have finite values
            for i in 0..n {
                let val = w.get_1d(i)?;
                assert!(val.is_finite());

                // Some windows can have negative values
                // Only check non-negativity for windows that should be non-negative
                match window_type {
                    Window::Cosine => {
                        // Cosine window can have negative values, just check range
                        assert!(
                            val >= -1.0 && val <= 1.0,
                            "Cosine window value {} out of range",
                            val
                        );
                    }
                    Window::Exponential(_) => {
                        // Exponential window should be non-negative but can be very small
                        assert!(
                            val >= 0.0,
                            "Exponential window should be non-negative, got {}",
                            val
                        );
                    }
                    _ => {
                        // Most other windows should be non-negative
                        assert!(
                            val >= -1e-6,
                            "Window {:?} value {} should be non-negative (allowing small numerical errors)",
                            window_type,
                            val
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_edge_cases() -> Result<()> {
        // Test zero-length windows
        let empty = hamming_window(0, false)?;
        assert_eq!(empty.shape().dims(), &[0]);

        // Test single-point windows
        let single = hamming_window(1, false)?;
        assert_eq!(single.shape().dims(), &[1]);
        assert_relative_eq!(single.get_1d(0)?, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_window_properties() -> Result<()> {
        let n = 16;

        // Test that rectangular window is all ones
        let rect = rectangular_window(n)?;
        for i in 0..n {
            assert_relative_eq!(rect.get_1d(i)?, 1.0, epsilon = 1e-6);
        }

        // Test that Tukey with alpha=0 equals rectangular
        let tukey_rect = tukey_window(n, 0.0, false)?;
        let diff = rect.sub(&tukey_rect)?.abs()?.sum()?;
        assert!(diff.item().unwrap() < 1e-6);

        // Test that Tukey with alpha=1 equals Hann
        let tukey_hann = tukey_window(n, 1.0, false)?;
        let hann = hann_window(n, false)?;
        let diff = hann.sub(&tukey_hann)?.abs()?.sum()?;
        assert!(diff.item().unwrap() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_modified_bessel_i0() {
        // Test some known values of the modified Bessel function
        assert_relative_eq!(modified_bessel_i0(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(modified_bessel_i0(1.0), 1.2660658777520084, epsilon = 1e-10);
        assert_relative_eq!(modified_bessel_i0(2.0), 2.2795853023360673, epsilon = 1e-10);
    }

    #[test]
    fn test_power_normalization() -> Result<()> {
        let window = hamming_window(10, false)?;
        let normalized = normalize_window(&window, "power")?;

        // Power normalization: sum of squares should be 1
        let power_sum = normalized.pow_scalar(2.0)?.sum()?;
        assert_relative_eq!(power_sum.item().unwrap(), 1.0, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_invalid_normalization() {
        let window = hamming_window(10, false).unwrap();
        let result = normalize_window(&window, "invalid");
        assert!(result.is_err());
    }
}
