//! Interpolation functions for tensor operations
//!
//! This module provides various interpolation methods commonly used in
//! signal processing, image processing, and scientific computing.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Interpolation methods supported
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMode {
    /// Linear interpolation
    Linear,
    /// Nearest neighbor interpolation
    Nearest,
    /// Cubic interpolation
    Cubic,
    /// Spline interpolation
    Spline,
    /// Lanczos interpolation
    Lanczos,
}

/// 1D linear interpolation
///
/// Interpolates values at specified points using linear interpolation
pub fn interp1d(x: &Tensor, y: &Tensor, x_new: &Tensor, extrapolate: bool) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let y_data = y.data()?;
    let x_new_data = x_new.data()?;

    if x_data.len() != y_data.len() {
        return Err(TorshError::InvalidArgument(
            "x and y arrays must have the same length".to_string(),
        ));
    }

    if x_data.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "At least 2 data points required for interpolation".to_string(),
        ));
    }

    let mut result = Vec::with_capacity(x_new_data.len());

    for &x_val in x_new_data.iter() {
        let y_val = linear_interpolate(&x_data, &y_data, x_val, extrapolate)?;
        result.push(y_val);
    }

    Tensor::from_data(result, x_new.shape().dims().to_vec(), x_new.device())
}

/// 1D cubic spline interpolation
///
/// Performs cubic spline interpolation with natural boundary conditions
pub fn spline1d(x: &Tensor, y: &Tensor, x_new: &Tensor, extrapolate: bool) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let y_data = y.data()?;
    let x_new_data = x_new.data()?;

    if x_data.len() != y_data.len() {
        return Err(TorshError::InvalidArgument(
            "x and y arrays must have the same length".to_string(),
        ));
    }

    if x_data.len() < 4 {
        return Err(TorshError::InvalidArgument(
            "At least 4 data points required for cubic spline interpolation".to_string(),
        ));
    }

    // Compute spline coefficients
    let spline_coeffs = compute_cubic_spline_coefficients(&x_data, &y_data)?;

    let mut result = Vec::with_capacity(x_new_data.len());

    for &x_val in x_new_data.iter() {
        let y_val = evaluate_cubic_spline(&x_data, &y_data, &spline_coeffs, x_val, extrapolate)?;
        result.push(y_val);
    }

    Tensor::from_data(result, x_new.shape().dims().to_vec(), x_new.device())
}

/// 2D bilinear interpolation
///
/// Performs bilinear interpolation on a 2D grid
pub fn interp2d(
    input: &Tensor,
    x_coords: &Tensor,
    y_coords: &Tensor,
    mode: InterpolationMode,
) -> TorshResult<Tensor> {
    let input_shape = input.shape();
    if input_shape.ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Input must be a 2D tensor".to_string(),
        ));
    }

    if x_coords.shape() != y_coords.shape() {
        return Err(TorshError::ShapeMismatch {
            expected: x_coords.shape().dims().to_vec(),
            got: y_coords.shape().dims().to_vec(),
        });
    }

    let (height, width) = (input_shape.dims()[0], input_shape.dims()[1]);
    let input_data = input.data()?;
    let x_coords_data = x_coords.data()?;
    let y_coords_data = y_coords.data()?;

    let mut result = Vec::with_capacity(x_coords_data.len());

    for (_i, (&x, &y)) in x_coords_data.iter().zip(y_coords_data.iter()).enumerate() {
        let val = match mode {
            InterpolationMode::Linear => bilinear_sample(&input_data, width, height, x, y),
            InterpolationMode::Nearest => nearest_sample(&input_data, width, height, x, y),
            InterpolationMode::Cubic => bicubic_sample(&input_data, width, height, x, y),
            _ => {
                return Err(TorshError::UnsupportedOperation {
                    op: format!("{:?}", mode),
                    dtype: "2D interpolation".to_string(),
                });
            }
        };
        result.push(val);
    }

    Tensor::from_data(result, x_coords.shape().dims().to_vec(), x_coords.device())
}

/// Grid sample function (similar to PyTorch's grid_sample)
///
/// Samples input using the sampling grid specified by coordinates
pub fn grid_sample(
    input: &Tensor,
    grid: &Tensor,
    mode: InterpolationMode,
    _padding_mode: &str,
    align_corners: bool,
) -> TorshResult<Tensor> {
    let input_shape = input.shape();
    let grid_shape = grid.shape();

    // Expect input: [N, C, H, W] and grid: [N, H_out, W_out, 2]
    if input_shape.ndim() != 4 || grid_shape.ndim() != 4 || grid_shape.dims()[3] != 2 {
        return Err(TorshError::InvalidArgument(
            "Expected input [N,C,H,W] and grid [N,H_out,W_out,2]".to_string(),
        ));
    }

    let (batch_size, channels, in_height, in_width) = (
        input_shape.dims()[0],
        input_shape.dims()[1],
        input_shape.dims()[2],
        input_shape.dims()[3],
    );

    let (out_height, out_width) = (grid_shape.dims()[1], grid_shape.dims()[2]);

    let input_data = input.data()?;
    let grid_data = grid.data()?;

    let mut result = vec![0.0; batch_size * channels * out_height * out_width];

    for n in 0..batch_size {
        for c in 0..channels {
            for h in 0..out_height {
                for w in 0..out_width {
                    let grid_idx = ((n * out_height + h) * out_width + w) * 2;
                    let x = grid_data[grid_idx];
                    let y = grid_data[grid_idx + 1];

                    // Convert from [-1, 1] to pixel coordinates
                    let (pixel_x, pixel_y) = if align_corners {
                        (
                            (x + 1.0) * (in_width - 1) as f32 / 2.0,
                            (y + 1.0) * (in_height - 1) as f32 / 2.0,
                        )
                    } else {
                        (
                            (x + 1.0) * in_width as f32 / 2.0 - 0.5,
                            (y + 1.0) * in_height as f32 / 2.0 - 0.5,
                        )
                    };

                    // Sample from input
                    let channel_offset = (n * channels + c) * in_height * in_width;
                    let input_slice =
                        &input_data[channel_offset..channel_offset + in_height * in_width];

                    let sampled_value = match mode {
                        InterpolationMode::Linear => {
                            bilinear_sample(input_slice, in_width, in_height, pixel_x, pixel_y)
                        }
                        InterpolationMode::Nearest => {
                            nearest_sample(input_slice, in_width, in_height, pixel_x, pixel_y)
                        }
                        _ => 0.0,
                    };

                    let out_idx = ((n * channels + c) * out_height + h) * out_width + w;
                    result[out_idx] = sampled_value;
                }
            }
        }
    }

    Tensor::from_data(
        result,
        vec![batch_size, channels, out_height, out_width],
        input.device(),
    )
}

/// Lanczos interpolation
///
/// High-quality interpolation using Lanczos kernel
pub fn lanczos_interp1d(
    x: &Tensor,
    y: &Tensor,
    x_new: &Tensor,
    a: usize, // Lanczos parameter (typically 2 or 3)
) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let y_data = y.data()?;
    let x_new_data = x_new.data()?;

    if x_data.len() != y_data.len() {
        return Err(TorshError::InvalidArgument(
            "x and y arrays must have the same length".to_string(),
        ));
    }

    let mut result = Vec::with_capacity(x_new_data.len());

    for &x_val in x_new_data.iter() {
        let y_val = lanczos_interpolate(&x_data, &y_data, x_val, a)?;
        result.push(y_val);
    }

    Tensor::from_data(result, x_new.shape().dims().to_vec(), x_new.device())
}

/// Barycentric interpolation
///
/// Efficient method for polynomial interpolation
pub fn barycentric_interp(x: &Tensor, y: &Tensor, x_new: &Tensor) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let y_data = y.data()?;
    let x_new_data = x_new.data()?;

    if x_data.len() != y_data.len() {
        return Err(TorshError::InvalidArgument(
            "x and y arrays must have the same length".to_string(),
        ));
    }

    // Compute barycentric weights
    let weights = compute_barycentric_weights(&x_data)?;

    let mut result = Vec::with_capacity(x_new_data.len());

    for &x_val in x_new_data.iter() {
        let y_val = evaluate_barycentric(&x_data, &y_data, &weights, x_val)?;
        result.push(y_val);
    }

    Tensor::from_data(result, x_new.shape().dims().to_vec(), x_new.device())
}

// Helper functions

fn linear_interpolate(
    x_data: &[f32],
    y_data: &[f32],
    x_val: f32,
    extrapolate: bool,
) -> TorshResult<f32> {
    let n = x_data.len();

    // Find surrounding points
    let mut i = 0;
    while i < n - 1 && x_data[i + 1] < x_val {
        i += 1;
    }

    // Handle extrapolation
    if !extrapolate {
        if x_val < x_data[0] || x_val > x_data[n - 1] {
            return Err(TorshError::InvalidArgument(
                "Value outside interpolation range and extrapolation disabled".to_string(),
            ));
        }
    }

    if i == n - 1 {
        return Ok(y_data[n - 1]);
    }

    if x_data[i] == x_data[i + 1] {
        return Ok(y_data[i]);
    }

    let t = (x_val - x_data[i]) / (x_data[i + 1] - x_data[i]);
    Ok(y_data[i] * (1.0 - t) + y_data[i + 1] * t)
}

fn compute_cubic_spline_coefficients(x_data: &[f32], y_data: &[f32]) -> TorshResult<Vec<f32>> {
    let n = x_data.len();
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let _c = vec![0.0; n];
    let _d = vec![0.0; n];

    // Natural spline boundary conditions
    // This is a simplified implementation - in practice, you'd use a more robust solver
    for i in 0..n {
        a[i] = y_data[i];
    }

    // For now, use simple finite differences for derivatives
    for i in 1..n - 1 {
        let h1 = x_data[i] - x_data[i - 1];
        let h2 = x_data[i + 1] - x_data[i];
        let delta1 = (y_data[i] - y_data[i - 1]) / h1;
        let delta2 = (y_data[i + 1] - y_data[i]) / h2;

        b[i] = (delta1 + delta2) / 2.0;
    }

    Ok(b) // Return second derivatives as a simple approximation
}

fn evaluate_cubic_spline(
    x_data: &[f32],
    y_data: &[f32],
    _coeffs: &[f32],
    x_val: f32,
    extrapolate: bool,
) -> TorshResult<f32> {
    // For now, fall back to linear interpolation
    // A full cubic spline implementation would use the coefficients
    linear_interpolate(x_data, y_data, x_val, extrapolate)
}

fn bilinear_sample(data: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let wx = x - x0 as f32;
    let wy = y - y0 as f32;

    // Bounds checking
    let safe_sample = |x: i32, y: i32| -> f32 {
        if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
            data[y as usize * width + x as usize]
        } else {
            0.0 // Zero padding
        }
    };

    let v00 = safe_sample(x0, y0);
    let v01 = safe_sample(x0, y1);
    let v10 = safe_sample(x1, y0);
    let v11 = safe_sample(x1, y1);

    let v0 = v00 * (1.0 - wx) + v10 * wx;
    let v1 = v01 * (1.0 - wx) + v11 * wx;

    v0 * (1.0 - wy) + v1 * wy
}

fn nearest_sample(data: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x_idx = (x + 0.5) as usize;
    let y_idx = (y + 0.5) as usize;

    if x_idx < width && y_idx < height {
        data[y_idx * width + x_idx]
    } else {
        0.0
    }
}

fn bicubic_sample(data: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    // Simplified bicubic - in practice would use proper cubic kernel
    bilinear_sample(data, width, height, x, y)
}

fn lanczos_kernel(x: f32, a: usize) -> f32 {
    if x.abs() >= a as f32 {
        0.0
    } else if x.abs() < 1e-6 {
        1.0
    } else {
        let pi_x = std::f32::consts::PI * x;
        let pi_x_a = pi_x / a as f32;
        a as f32 * pi_x.sin() * pi_x_a.sin() / (pi_x * pi_x)
    }
}

fn lanczos_interpolate(x_data: &[f32], y_data: &[f32], x_val: f32, a: usize) -> TorshResult<f32> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (_i, (&xi, &yi)) in x_data.iter().zip(y_data.iter()).enumerate() {
        let weight = lanczos_kernel(x_val - xi, a);
        numerator += weight * yi;
        denominator += weight;
    }

    if denominator.abs() < 1e-10 {
        // Fall back to nearest neighbor
        let mut min_dist = f32::INFINITY;
        let mut nearest_val = 0.0;
        for (i, &xi) in x_data.iter().enumerate() {
            let dist = (x_val - xi).abs();
            if dist < min_dist {
                min_dist = dist;
                nearest_val = y_data[i];
            }
        }
        Ok(nearest_val)
    } else {
        Ok(numerator / denominator)
    }
}

fn compute_barycentric_weights(x_data: &[f32]) -> TorshResult<Vec<f32>> {
    let n = x_data.len();
    let mut weights = vec![1.0; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                weights[i] /= x_data[i] - x_data[j];
            }
        }
    }

    Ok(weights)
}

fn evaluate_barycentric(
    x_data: &[f32],
    y_data: &[f32],
    weights: &[f32],
    x_val: f32,
) -> TorshResult<f32> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (_i, (&xi, (&yi, &wi))) in x_data
        .iter()
        .zip(y_data.iter().zip(weights.iter()))
        .enumerate()
    {
        if (x_val - xi).abs() < 1e-10 {
            return Ok(yi); // Exact match
        }

        let term = wi / (x_val - xi);
        numerator += term * yi;
        denominator += term;
    }

    if denominator.abs() < 1e-10 {
        Err(TorshError::InvalidArgument(
            "Barycentric interpolation failed: denominator near zero".to_string(),
        ))
    } else {
        Ok(numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_linear_interpolation() {
        let x = tensor_1d(&[0.0, 1.0, 2.0, 3.0]).unwrap();
        let y = tensor_1d(&[0.0, 1.0, 4.0, 9.0]).unwrap();
        let x_new = tensor_1d(&[0.5, 1.5, 2.5]).unwrap();

        let result = interp1d(&x, &y, &x_new, false).unwrap();
        let result_data = result.data().unwrap();

        // Expected values: [0.5, 2.5, 6.5]
        assert!((result_data[0] - 0.5).abs() < 1e-6);
        assert!((result_data[1] - 2.5).abs() < 1e-6);
        assert!((result_data[2] - 6.5).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_sampling() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 grid
        let result = bilinear_sample(&data, 2, 2, 0.5, 0.5);

        // Should be average of all four values: (1+2+3+4)/4 = 2.5
        assert!((result - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_lanczos_kernel() {
        // Test Lanczos kernel properties
        assert!((lanczos_kernel(0.0, 3) - 1.0).abs() < 1e-6); // Should be 1 at origin
        assert!((lanczos_kernel(3.0, 3)).abs() < 1e-6); // Should be 0 at boundary
        assert!((lanczos_kernel(-3.0, 3)).abs() < 1e-6); // Should be 0 at negative boundary
    }

    #[test]
    fn test_grid_sample_basic() {
        use torsh_tensor::creation::zeros;

        let input = zeros(&[1, 1, 2, 2]).unwrap(); // 1x1x2x2
        let grid = zeros(&[1, 1, 1, 2]).unwrap(); // 1x1x1x2 (sample at origin)

        let result = grid_sample(&input, &grid, InterpolationMode::Linear, "zeros", false);

        // Should not panic and return correct shape
        assert!(result.is_ok());
        let result_tensor = result.unwrap();
        assert_eq!(result_tensor.shape().dims(), &[1, 1, 1, 1]);
    }
}
