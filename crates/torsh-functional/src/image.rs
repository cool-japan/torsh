//! Image processing operations

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Resize tensor using different interpolation methods
pub fn resize(
    input: &Tensor,
    size: (usize, usize),
    mode: InterpolationMode,
    antialias: bool,
) -> TorshResult<Tensor> {
    let shape = input.shape();
    if shape.ndim() < 3 {
        return Err(TorshError::invalid_argument_with_context(
            "Input tensor must have at least 3 dimensions (C, H, W)",
            "resize",
        ));
    }

    let dims = shape.dims();
    let channels = dims[dims.len() - 3];
    let in_height = dims[dims.len() - 2];
    let in_width = dims[dims.len() - 1];
    let (out_height, out_width) = size;

    // Handle batch dimensions
    let batch_dims: Vec<usize> = dims[..dims.len() - 3].to_vec();
    let batch_size = batch_dims.iter().product::<usize>();

    let input_data = input.to_vec()?;
    let mut output_data = vec![0.0f32; batch_size * channels * out_height * out_width];

    let scale_h = in_height as f32 / out_height as f32;
    let scale_w = in_width as f32 / out_width as f32;

    match mode {
        InterpolationMode::Nearest => {
            for b in 0..batch_size {
                for c in 0..channels {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let ih = (oh as f32 * scale_h).round() as usize;
                            let iw = (ow as f32 * scale_w).round() as usize;

                            let ih = ih.min(in_height - 1);
                            let iw = iw.min(in_width - 1);

                            let in_idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                            let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;

                            output_data[out_idx] = input_data[in_idx];
                        }
                    }
                }
            }
        }
        InterpolationMode::Bilinear => {
            for b in 0..batch_size {
                for c in 0..channels {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let fh = (oh as f32 + 0.5) * scale_h - 0.5;
                            let fw = (ow as f32 + 0.5) * scale_w - 0.5;

                            let ih_low = fh.floor() as i32;
                            let iw_low = fw.floor() as i32;
                            let _ih_high = ih_low + 1;
                            let _iw_high = iw_low + 1;

                            let wh = fh - ih_low as f32;
                            let ww = fw - iw_low as f32;

                            let mut value = 0.0f32;

                            // Bilinear interpolation
                            for dh in 0..2 {
                                for dw in 0..2 {
                                    let ih = ih_low + dh;
                                    let iw = iw_low + dw;

                                    if ih >= 0
                                        && ih < in_height as i32
                                        && iw >= 0
                                        && iw < in_width as i32
                                    {
                                        let weight = if dh == 0 { 1.0 - wh } else { wh }
                                            * if dw == 0 { 1.0 - ww } else { ww };

                                        let in_idx = ((b * channels + c) * in_height + ih as usize)
                                            * in_width
                                            + iw as usize;
                                        value += weight * input_data[in_idx];
                                    }
                                }
                            }

                            let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                            output_data[out_idx] = value;
                        }
                    }
                }
            }
        }
        InterpolationMode::Bicubic | InterpolationMode::Area => {
            // Simplified implementation - use bilinear for now
            return resize(input, size, InterpolationMode::Bilinear, antialias);
        }
    }

    let mut output_shape = batch_dims;
    output_shape.extend_from_slice(&[channels, out_height, out_width]);

    Tensor::from_data(output_data, output_shape, input.device())
}

/// Interpolation modes for resizing
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMode {
    Nearest,
    Bilinear,
    Bicubic,
    Area,
}

/// Apply Gaussian blur to image tensor
pub fn gaussian_blur(input: &Tensor, kernel_size: usize, sigma: f32) -> TorshResult<Tensor> {
    let shape = input.shape();
    if shape.ndim() < 3 {
        return Err(TorshError::invalid_argument_with_context(
            "Input tensor must have at least 3 dimensions (C, H, W)",
            "gaussian_blur",
        ));
    }

    // Create Gaussian kernel
    let radius = kernel_size / 2;
    let mut kernel = vec![0.0f32; kernel_size * kernel_size];
    let mut sum = 0.0f32;

    for i in 0..kernel_size {
        for j in 0..kernel_size {
            let x = i as i32 - radius as i32;
            let y = j as i32 - radius as i32;
            let val = (-((x * x + y * y) as f32) / (2.0 * sigma * sigma)).exp();
            kernel[i * kernel_size + j] = val;
            sum += val;
        }
    }

    // Normalize kernel
    for val in &mut kernel {
        *val /= sum;
    }

    // Apply convolution with the Gaussian kernel
    apply_convolution(input, &kernel, kernel_size, 1, radius)
}

/// Apply Sobel edge detection
pub fn sobel_filter(input: &Tensor, direction: SobelDirection) -> TorshResult<Tensor> {
    let kernel = match direction {
        SobelDirection::X => vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
        SobelDirection::Y => vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
        SobelDirection::Both => {
            // For both directions, compute magnitude
            let x_result = sobel_filter(input, SobelDirection::X)?;
            let y_result = sobel_filter(input, SobelDirection::Y)?;
            return compute_gradient_magnitude(&x_result, &y_result);
        }
    };

    apply_convolution(input, &kernel, 3, 1, 1)
}

/// Sobel filter directions
#[derive(Debug, Clone, Copy)]
pub enum SobelDirection {
    X,
    Y,
    Both,
}

/// Apply Laplacian filter for edge detection
pub fn laplacian_filter(input: &Tensor) -> TorshResult<Tensor> {
    let kernel = vec![0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0];

    apply_convolution(input, &kernel, 3, 1, 1)
}

/// Apply morphological erosion
pub fn erosion(input: &Tensor, kernel_size: usize, iterations: usize) -> TorshResult<Tensor> {
    let mut result = input.clone();

    for _ in 0..iterations {
        result = apply_morphological_op(&result, kernel_size, MorphOp::Erosion)?;
    }

    Ok(result)
}

/// Apply morphological dilation
pub fn dilation(input: &Tensor, kernel_size: usize, iterations: usize) -> TorshResult<Tensor> {
    let mut result = input.clone();

    for _ in 0..iterations {
        result = apply_morphological_op(&result, kernel_size, MorphOp::Dilation)?;
    }

    Ok(result)
}

/// Apply morphological opening (erosion followed by dilation)
pub fn opening(input: &Tensor, kernel_size: usize) -> TorshResult<Tensor> {
    let eroded = erosion(input, kernel_size, 1)?;
    dilation(&eroded, kernel_size, 1)
}

/// Apply morphological closing (dilation followed by erosion)
pub fn closing(input: &Tensor, kernel_size: usize) -> TorshResult<Tensor> {
    let dilated = dilation(input, kernel_size, 1)?;
    erosion(&dilated, kernel_size, 1)
}

/// Convert RGB to HSV color space
pub fn rgb_to_hsv(input: &Tensor) -> TorshResult<Tensor> {
    let shape = input.shape();
    if shape.ndim() < 3 {
        return Err(TorshError::invalid_argument_with_context(
            "Input tensor must have at least 3 dimensions (C, H, W)",
            "rgb_to_hsv",
        ));
    }

    let dims = shape.dims();
    if dims[dims.len() - 3] != 3 {
        return Err(TorshError::invalid_argument_with_context(
            "Input tensor must have 3 channels for RGB",
            "rgb_to_hsv",
        ));
    }

    let input_data = input.to_vec()?;
    let mut output_data = vec![0.0f32; input_data.len()];

    let batch_size = dims[..dims.len() - 3].iter().product::<usize>();
    let height = dims[dims.len() - 2];
    let width = dims[dims.len() - 1];

    for b in 0..batch_size {
        for h in 0..height {
            for w in 0..width {
                let r_idx = ((b * 3 + 0) * height + h) * width + w;
                let g_idx = ((b * 3 + 1) * height + h) * width + w;
                let b_idx = ((b * 3 + 2) * height + h) * width + w;

                let r = input_data[r_idx];
                let g = input_data[g_idx];
                let b_val = input_data[b_idx];

                let max_val = r.max(g).max(b_val);
                let min_val = r.min(g).min(b_val);
                let delta = max_val - min_val;

                // Value
                let v = max_val;

                // Saturation
                let s = if max_val == 0.0 { 0.0 } else { delta / max_val };

                // Hue
                let h_val = if delta == 0.0 {
                    0.0
                } else if max_val == r {
                    60.0 * (((g - b_val) / delta) % 6.0)
                } else if max_val == g {
                    60.0 * ((b_val - r) / delta + 2.0)
                } else {
                    60.0 * ((r - g) / delta + 4.0)
                };

                output_data[r_idx] = h_val / 360.0; // Normalize hue to [0, 1]
                output_data[g_idx] = s;
                output_data[b_idx] = v;
            }
        }
    }

    Tensor::from_data(output_data, dims.to_vec(), input.device())
}

/// Convert HSV to RGB color space
pub fn hsv_to_rgb(input: &Tensor) -> TorshResult<Tensor> {
    let shape = input.shape();
    if shape.ndim() < 3 {
        return Err(TorshError::invalid_argument_with_context(
            "Input tensor must have at least 3 dimensions (C, H, W)",
            "hsv_to_rgb",
        ));
    }

    let dims = shape.dims();
    if dims[dims.len() - 3] != 3 {
        return Err(TorshError::invalid_argument_with_context(
            "Input tensor must have 3 channels for HSV",
            "hsv_to_rgb",
        ));
    }

    let input_data = input.to_vec()?;
    let mut output_data = vec![0.0f32; input_data.len()];

    let batch_size = dims[..dims.len() - 3].iter().product::<usize>();
    let height = dims[dims.len() - 2];
    let width = dims[dims.len() - 1];

    for b in 0..batch_size {
        for h in 0..height {
            for w in 0..width {
                let h_idx = ((b * 3 + 0) * height + h) * width + w;
                let s_idx = ((b * 3 + 1) * height + h) * width + w;
                let v_idx = ((b * 3 + 2) * height + h) * width + w;

                let h_val = input_data[h_idx] * 360.0; // Denormalize hue
                let s = input_data[s_idx];
                let v = input_data[v_idx];

                let c = v * s;
                let x = c * (1.0 - ((h_val / 60.0) % 2.0 - 1.0).abs());
                let m = v - c;

                let (r_prime, g_prime, b_prime) = if h_val < 60.0 {
                    (c, x, 0.0)
                } else if h_val < 120.0 {
                    (x, c, 0.0)
                } else if h_val < 180.0 {
                    (0.0, c, x)
                } else if h_val < 240.0 {
                    (0.0, x, c)
                } else if h_val < 300.0 {
                    (x, 0.0, c)
                } else {
                    (c, 0.0, x)
                };

                output_data[h_idx] = r_prime + m;
                output_data[s_idx] = g_prime + m;
                output_data[v_idx] = b_prime + m;
            }
        }
    }

    Tensor::from_data(output_data, dims.to_vec(), input.device())
}

/// Apply affine transformation to image
pub fn affine_transform(
    input: &Tensor,
    matrix: &[f32; 6], // [a, b, c, d, e, f] for transformation [[a, b, c], [d, e, f]]
    output_size: Option<(usize, usize)>,
    fill_value: f32,
) -> TorshResult<Tensor> {
    let shape = input.shape();
    if shape.ndim() < 3 {
        return Err(TorshError::invalid_argument_with_context(
            "Input tensor must have at least 3 dimensions (C, H, W)",
            "affine_transform",
        ));
    }

    let dims = shape.dims();
    let channels = dims[dims.len() - 3];
    let in_height = dims[dims.len() - 2];
    let in_width = dims[dims.len() - 1];

    let (out_height, out_width) = output_size.unwrap_or((in_height, in_width));

    // Compute inverse transformation matrix for backward mapping
    let [a, b, c, d, e, f] = *matrix;
    let det = a * e - b * d;

    if det.abs() < 1e-6 {
        return Err(TorshError::invalid_argument_with_context(
            "Affine transformation matrix is singular",
            "affine_transform",
        ));
    }

    let inv_det = 1.0 / det;
    let inv_a = e * inv_det;
    let inv_b = -b * inv_det;
    let inv_c = (b * f - c * e) * inv_det;
    let inv_d = -d * inv_det;
    let inv_e = a * inv_det;
    let inv_f = (c * d - a * f) * inv_det;

    let batch_dims: Vec<usize> = dims[..dims.len() - 3].to_vec();
    let batch_size = batch_dims.iter().product::<usize>();

    let input_data = input.to_vec()?;
    let mut output_data = vec![fill_value; batch_size * channels * out_height * out_width];

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    // Apply inverse transformation
                    let x_out = ow as f32;
                    let y_out = oh as f32;

                    let x_in = inv_a * x_out + inv_b * y_out + inv_c;
                    let y_in = inv_d * x_out + inv_e * y_out + inv_f;

                    // Bilinear interpolation
                    if x_in >= 0.0
                        && x_in < in_width as f32 - 1.0
                        && y_in >= 0.0
                        && y_in < in_height as f32 - 1.0
                    {
                        let x0 = x_in.floor() as usize;
                        let y0 = y_in.floor() as usize;
                        let x1 = x0 + 1;
                        let y1 = y0 + 1;

                        let wx = x_in - x0 as f32;
                        let wy = y_in - y0 as f32;

                        let idx00 = ((b * channels + c) * in_height + y0) * in_width + x0;
                        let idx01 = ((b * channels + c) * in_height + y0) * in_width + x1;
                        let idx10 = ((b * channels + c) * in_height + y1) * in_width + x0;
                        let idx11 = ((b * channels + c) * in_height + y1) * in_width + x1;

                        let val = (1.0 - wx) * (1.0 - wy) * input_data[idx00]
                            + wx * (1.0 - wy) * input_data[idx01]
                            + (1.0 - wx) * wy * input_data[idx10]
                            + wx * wy * input_data[idx11];

                        let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                        output_data[out_idx] = val;
                    }
                }
            }
        }
    }

    let mut output_shape = batch_dims;
    output_shape.extend_from_slice(&[channels, out_height, out_width]);

    Tensor::from_data(output_data, output_shape, input.device())
}

// Helper functions

/// Apply convolution with a given kernel
fn apply_convolution(
    input: &Tensor,
    kernel: &[f32],
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> TorshResult<Tensor> {
    let shape = input.shape();
    let dims = shape.dims();

    let batch_dims: Vec<usize> = dims[..dims.len() - 3].to_vec();
    let batch_size = batch_dims.iter().product::<usize>();
    let channels = dims[dims.len() - 3];
    let in_height = dims[dims.len() - 2];
    let in_width = dims[dims.len() - 1];

    let out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    let out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    let input_data = input.to_vec()?;
    let mut output_data = vec![0.0f32; batch_size * channels * out_height * out_width];

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = 0.0f32;

                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;

                            if ih >= padding
                                && ih < in_height + padding
                                && iw >= padding
                                && iw < in_width + padding
                            {
                                let real_ih = ih - padding;
                                let real_iw = iw - padding;

                                if real_ih < in_height && real_iw < in_width {
                                    let in_idx = ((b * channels + c) * in_height + real_ih)
                                        * in_width
                                        + real_iw;
                                    let kernel_idx = kh * kernel_size + kw;
                                    sum += input_data[in_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }

                    let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    output_data[out_idx] = sum;
                }
            }
        }
    }

    let mut output_shape = batch_dims;
    output_shape.extend_from_slice(&[channels, out_height, out_width]);

    Tensor::from_data(output_data, output_shape, input.device())
}

/// Morphological operation types
#[derive(Debug, Clone, Copy)]
enum MorphOp {
    Erosion,
    Dilation,
}

/// Apply morphological operation
fn apply_morphological_op(input: &Tensor, kernel_size: usize, op: MorphOp) -> TorshResult<Tensor> {
    let shape = input.shape();
    let dims = shape.dims();

    let batch_dims: Vec<usize> = dims[..dims.len() - 3].to_vec();
    let batch_size = batch_dims.iter().product::<usize>();
    let channels = dims[dims.len() - 3];
    let height = dims[dims.len() - 2];
    let width = dims[dims.len() - 1];

    let radius = kernel_size / 2;
    let input_data = input.to_vec()?;
    let mut output_data = vec![0.0f32; input_data.len()];

    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let mut result = match op {
                        MorphOp::Erosion => f32::INFINITY,
                        MorphOp::Dilation => f32::NEG_INFINITY,
                    };

                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = h as i32 + kh as i32 - radius as i32;
                            let iw = w as i32 + kw as i32 - radius as i32;

                            if ih >= 0 && ih < height as i32 && iw >= 0 && iw < width as i32 {
                                let in_idx = ((b * channels + c) * height + ih as usize) * width
                                    + iw as usize;
                                let val = input_data[in_idx];

                                match op {
                                    MorphOp::Erosion => result = result.min(val),
                                    MorphOp::Dilation => result = result.max(val),
                                }
                            }
                        }
                    }

                    let out_idx = ((b * channels + c) * height + h) * width + w;
                    output_data[out_idx] = result;
                }
            }
        }
    }

    Tensor::from_data(output_data, dims.to_vec(), input.device())
}

/// Compute gradient magnitude from X and Y gradients
fn compute_gradient_magnitude(grad_x: &Tensor, grad_y: &Tensor) -> TorshResult<Tensor> {
    let grad_x_data = grad_x.to_vec()?;
    let grad_y_data = grad_y.to_vec()?;

    let magnitude_data: Vec<f32> = grad_x_data
        .iter()
        .zip(grad_y_data.iter())
        .map(|(&gx, &gy)| (gx * gx + gy * gy).sqrt())
        .collect();

    Tensor::from_data(
        magnitude_data,
        grad_x.shape().dims().to_vec(),
        grad_x.device(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_resize_nearest() {
        let input = ones(&[1, 3, 4, 4]).unwrap(); // Batch=1, Channels=3, Height=4, Width=4
        let result = resize(&input, (2, 2), InterpolationMode::Nearest, false).unwrap();
        assert_eq!(result.shape().dims(), &[1, 3, 2, 2]);
    }

    #[test]
    fn test_gaussian_blur() {
        let input = ones(&[1, 3, 5, 5]).unwrap();
        let result = gaussian_blur(&input, 3, 1.0).unwrap();
        assert_eq!(result.shape().dims(), &[1, 3, 5, 5]); // Size maintained due to padding
    }

    #[test]
    fn test_sobel_filter() {
        let input = ones(&[1, 1, 5, 5]).unwrap();
        let result = sobel_filter(&input, SobelDirection::X).unwrap();
        assert_eq!(result.shape().dims(), &[1, 1, 5, 5]); // Size maintained due to padding
    }

    #[test]
    fn test_rgb_to_hsv_conversion() {
        let input = ones(&[1, 3, 2, 2]).unwrap(); // RGB image
        let hsv = rgb_to_hsv(&input).unwrap();
        let rgb_back = hsv_to_rgb(&hsv).unwrap();
        assert_eq!(rgb_back.shape().dims(), &[1, 3, 2, 2]);
    }

    #[test]
    fn test_morphological_operations() {
        let input = ones(&[1, 1, 5, 5]).unwrap();
        let eroded = erosion(&input, 3, 1).unwrap();
        let dilated = dilation(&input, 3, 1).unwrap();
        assert_eq!(eroded.shape().dims(), &[1, 1, 5, 5]);
        assert_eq!(dilated.shape().dims(), &[1, 1, 5, 5]);
    }
}
