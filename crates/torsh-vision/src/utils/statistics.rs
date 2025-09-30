//! Statistical Analysis Functions for Image Processing
//!
//! This module provides statistical analysis functions for image tensors and quality assessment metrics.
//! It includes functions for computing basic statistics like mean and standard deviation,
//! as well as advanced image quality metrics such as PSNR, SSIM, MSE, and MAE.

use crate::{Result, VisionError};
use image::DynamicImage;
use torsh_tensor::Tensor;

// Import image_to_tensor from parent module for calculate_stats function
use super::image_to_tensor;

/// Calculate image statistics (mean and std per channel)
///
/// Computes the mean and standard deviation for each color channel across a collection of images.
/// This is useful for dataset normalization and understanding data distribution.
///
/// # Arguments
/// * `images` - Slice of DynamicImage objects to analyze
///
/// # Returns
/// Tuple of (means, stds) where each vector contains values for RGB channels
///
/// # Example
/// ```
/// use torsh_vision::utils::statistics::calculate_stats;
/// use image::DynamicImage;
///
/// let images: Vec<DynamicImage> = load_dataset_images();
/// let (means, stds) = calculate_stats(&images)?;
/// println!("Channel means: {:?}", means);
/// println!("Channel stds: {:?}", stds);
/// ```
pub fn calculate_stats(images: &[DynamicImage]) -> Result<(Vec<f32>, Vec<f32>)> {
    if images.is_empty() {
        return Err(VisionError::InvalidArgument(
            "No images provided".to_string(),
        ));
    }

    // Convert images to tensors and calculate statistics
    let mut all_pixels: Vec<Vec<f32>> = vec![Vec::new(); 3]; // RGB channels

    for image in images {
        let tensor = image_to_tensor(image)?;
        let shape = tensor.shape();

        if shape.dims()[0] == 3 {
            // RGB image
            for c in 0..3 {
                for y in 0..shape.dims()[1] {
                    for x in 0..shape.dims()[2] {
                        let pixel_val = tensor.get(&[c, y, x])?;
                        all_pixels[c].push(pixel_val);
                    }
                }
            }
        } else if shape.dims()[0] == 1 {
            // Grayscale image - replicate across all channels
            for y in 0..shape.dims()[1] {
                for x in 0..shape.dims()[2] {
                    let pixel_val = tensor.get(&[0, y, x])?;
                    for c in 0..3 {
                        all_pixels[c].push(pixel_val);
                    }
                }
            }
        }
    }

    // Calculate mean and std for each channel
    let mut means = Vec::new();
    let mut stds = Vec::new();

    for channel_pixels in &all_pixels {
        if channel_pixels.is_empty() {
            means.push(0.0);
            stds.push(1.0);
            continue;
        }

        // Calculate mean
        let sum: f32 = channel_pixels.iter().sum();
        let mean = sum / channel_pixels.len() as f32;
        means.push(mean);

        // Calculate standard deviation
        let variance: f32 = channel_pixels
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / channel_pixels.len() as f32;
        let std = variance.sqrt();
        stds.push(std.max(1e-8)); // Avoid division by zero
    }

    Ok((means, stds))
}

/// Calculate Peak Signal-to-Noise Ratio (PSNR) between two images
///
/// PSNR is a metric used to measure the quality of a reconstruction of a lossy compression codec.
/// Higher PSNR values indicate better quality (less noise).
///
/// # Arguments
/// * `image1` - Reference image tensor (C, H, W)
/// * `image2` - Comparison image tensor (C, H, W)
/// * `max_val` - Maximum possible pixel value (default: 1.0 for normalized images)
///
/// # Returns
/// PSNR value in decibels (dB). Higher values indicate better quality.
///
/// # Example
/// ```
/// use torsh_vision::utils::statistics::psnr;
/// use torsh_tensor::Tensor;
///
/// let original_image: Tensor<f32> = load_reference_image();
/// let compressed_image: Tensor<f32> = load_compressed_image();
/// let psnr_value = psnr(&original_image, &compressed_image, Some(1.0))?;
/// println!("PSNR: {:.2} dB", psnr_value);
/// ```
pub fn psnr(image1: &Tensor<f32>, image2: &Tensor<f32>, max_val: Option<f32>) -> Result<f32> {
    // Input validation
    let shape1 = image1.shape();
    let shape2 = image2.shape();

    if shape1.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensor (C, H, W) for image1, got {}D",
            shape1.dims().len()
        )));
    }

    if shape2.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensor (C, H, W) for image2, got {}D",
            shape2.dims().len()
        )));
    }

    if shape1.dims() != shape2.dims() {
        return Err(VisionError::InvalidArgument(
            "Input tensors must have the same shape".to_string(),
        ));
    }

    let max_val = max_val.unwrap_or(1.0);

    // Calculate Mean Squared Error (MSE)
    let diff = image1
        .sub(image2)
        .map_err(|e| VisionError::TensorError(e))?;
    let squared_diff = diff.mul(&diff).map_err(|e| VisionError::TensorError(e))?;
    let mse = squared_diff.mean(None, false)?.item()?;

    // Avoid division by zero
    if mse < 1e-10 {
        return Ok(f32::INFINITY); // Images are identical
    }

    // Calculate PSNR
    let psnr_value = 20.0 * (max_val / mse.sqrt()).log10();
    Ok(psnr_value)
}

/// Calculate Structural Similarity Index (SSIM) between two images
///
/// SSIM is a perceptual metric that quantifies image quality degradation caused by processing.
/// SSIM values range from -1 to 1, where 1 indicates perfect structural similarity.
///
/// # Arguments
/// * `image1` - Reference image tensor (C, H, W)
/// * `image2` - Comparison image tensor (C, H, W)
/// * `window_size` - Size of the sliding window (default: 11)
/// * `k1` - Algorithm parameter (default: 0.01)
/// * `k2` - Algorithm parameter (default: 0.03)
///
/// # Returns
/// SSIM value between -1 and 1. Higher values indicate better structural similarity.
///
/// # Example
/// ```
/// use torsh_vision::utils::statistics::ssim;
/// use torsh_tensor::Tensor;
///
/// let original_image: Tensor<f32> = load_reference_image();
/// let processed_image: Tensor<f32> = load_processed_image();
/// let ssim_value = ssim(&original_image, &processed_image, None, None, None)?;
/// println!("SSIM: {:.4}", ssim_value);
/// ```
pub fn ssim(
    image1: &Tensor<f32>,
    image2: &Tensor<f32>,
    window_size: Option<usize>,
    k1: Option<f32>,
    k2: Option<f32>,
) -> Result<f32> {
    // Input validation
    let shape1 = image1.shape();
    let shape2 = image2.shape();

    if shape1.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensor (C, H, W) for image1, got {}D",
            shape1.dims().len()
        )));
    }

    if shape2.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensor (C, H, W) for image2, got {}D",
            shape2.dims().len()
        )));
    }

    if shape1.dims() != shape2.dims() {
        return Err(VisionError::InvalidArgument(
            "Input tensors must have the same shape".to_string(),
        ));
    }

    let window_size = window_size.unwrap_or(11);
    let k1 = k1.unwrap_or(0.01);
    let k2 = k2.unwrap_or(0.03);
    let data_range = 1.0; // Assuming normalized images

    let c1 = (k1 * data_range).powi(2);
    let c2 = (k2 * data_range).powi(2);

    let (channels, height, width) = (shape1.dims()[0], shape1.dims()[1], shape1.dims()[2]);

    // Check if window size is appropriate
    if window_size > height || window_size > width {
        return Err(VisionError::InvalidArgument(format!(
            "Window size ({}) too large for image dimensions ({}x{})",
            window_size, height, width
        )));
    }

    let mut ssim_total = 0.0;
    let mut valid_windows = 0;

    // Calculate SSIM for each channel
    for c in 0..channels {
        let mut channel_ssim = 0.0;
        let mut channel_windows = 0;

        // Sliding window approach
        for y in 0..=(height - window_size) {
            for x in 0..=(width - window_size) {
                // Extract windows
                let (mu1, mu2, sigma1_sq, sigma2_sq, sigma12) =
                    calculate_window_statistics(image1, image2, c, y, x, window_size)?;

                // Calculate SSIM for this window
                let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
                let denominator = (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2);

                if denominator > 0.0 {
                    channel_ssim += numerator / denominator;
                    channel_windows += 1;
                }
            }
        }

        if channel_windows > 0 {
            ssim_total += channel_ssim / channel_windows as f32;
            valid_windows += 1;
        }
    }

    if valid_windows > 0 {
        Ok(ssim_total / valid_windows as f32)
    } else {
        Ok(0.0)
    }
}

/// Helper function to calculate statistics for a window in SSIM computation
fn calculate_window_statistics(
    image1: &Tensor<f32>,
    image2: &Tensor<f32>,
    channel: usize,
    start_y: usize,
    start_x: usize,
    window_size: usize,
) -> Result<(f32, f32, f32, f32, f32)> {
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum1_sq = 0.0;
    let mut sum2_sq = 0.0;
    let mut sum12 = 0.0;
    let n = (window_size * window_size) as f32;

    // Calculate sums over the window
    for y in start_y..(start_y + window_size) {
        for x in start_x..(start_x + window_size) {
            let val1 = image1.get(&[channel, y, x])?;
            let val2 = image2.get(&[channel, y, x])?;

            sum1 += val1;
            sum2 += val2;
            sum1_sq += val1 * val1;
            sum2_sq += val2 * val2;
            sum12 += val1 * val2;
        }
    }

    // Calculate statistics
    let mu1 = sum1 / n;
    let mu2 = sum2 / n;
    let sigma1_sq = (sum1_sq / n) - (mu1 * mu1);
    let sigma2_sq = (sum2_sq / n) - (mu2 * mu2);
    let sigma12 = (sum12 / n) - (mu1 * mu2);

    Ok((mu1, mu2, sigma1_sq, sigma2_sq, sigma12))
}

/// Calculate Mean Squared Error (MSE) between two images
///
/// MSE is the average of the squared differences between corresponding pixels.
/// Lower values indicate better similarity.
///
/// # Arguments
/// * `image1` - Reference image tensor (C, H, W)
/// * `image2` - Comparison image tensor (C, H, W)
///
/// # Returns
/// MSE value. Lower values indicate better similarity.
///
/// # Example
/// ```
/// use torsh_vision::utils::statistics::mse;
/// use torsh_tensor::Tensor;
///
/// let image1: Tensor<f32> = load_image_tensor();
/// let image2: Tensor<f32> = load_image_tensor();
/// let mse_value = mse(&image1, &image2)?;
/// println!("MSE: {:.6}", mse_value);
/// ```
pub fn mse(image1: &Tensor<f32>, image2: &Tensor<f32>) -> Result<f32> {
    // Input validation
    let shape1 = image1.shape();
    let shape2 = image2.shape();

    if shape1.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensor (C, H, W) for image1, got {}D",
            shape1.dims().len()
        )));
    }

    if shape2.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensor (C, H, W) for image2, got {}D",
            shape2.dims().len()
        )));
    }

    if shape1.dims() != shape2.dims() {
        return Err(VisionError::InvalidArgument(
            "Input tensors must have the same shape".to_string(),
        ));
    }

    // Calculate MSE
    let diff = image1
        .sub(image2)
        .map_err(|e| VisionError::TensorError(e))?;
    let squared_diff = diff.mul(&diff).map_err(|e| VisionError::TensorError(e))?;
    let mse_value = squared_diff.mean(None, false)?.item()?;

    Ok(mse_value)
}

/// Calculate Mean Absolute Error (MAE) between two images
///
/// MAE is the average of the absolute differences between corresponding pixels.
/// Lower values indicate better similarity.
///
/// # Arguments
/// * `image1` - Reference image tensor (C, H, W)
/// * `image2` - Comparison image tensor (C, H, W)
///
/// # Returns
/// MAE value. Lower values indicate better similarity.
///
/// # Example
/// ```
/// use torsh_vision::utils::statistics::mae;
/// use torsh_tensor::Tensor;
///
/// let image1: Tensor<f32> = load_image_tensor();
/// let image2: Tensor<f32> = load_image_tensor();
/// let mae_value = mae(&image1, &image2)?;
/// println!("MAE: {:.6}", mae_value);
/// ```
pub fn mae(image1: &Tensor<f32>, image2: &Tensor<f32>) -> Result<f32> {
    // Input validation
    let shape1 = image1.shape();
    let shape2 = image2.shape();

    if shape1.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensor (C, H, W) for image1, got {}D",
            shape1.dims().len()
        )));
    }

    if shape2.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensor (C, H, W) for image2, got {}D",
            shape2.dims().len()
        )));
    }

    if shape1.dims() != shape2.dims() {
        return Err(VisionError::InvalidArgument(
            "Input tensors must have the same shape".to_string(),
        ));
    }

    // Calculate MAE
    let diff = image1
        .sub(image2)
        .map_err(|e| VisionError::TensorError(e))?;
    let abs_diff = diff.abs().map_err(|e| VisionError::TensorError(e))?;
    let mae_value = abs_diff.mean(None, false)?.item()?;

    Ok(mae_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_mse_identical_images() {
        let tensor = creation::ones(&[3, 32, 32]).unwrap();
        let result = mse(&tensor, &tensor).unwrap();
        assert!((result - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_mae_identical_images() {
        let tensor = creation::ones(&[3, 32, 32]).unwrap();
        let result = mae(&tensor, &tensor).unwrap();
        assert!((result - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_psnr_identical_images() {
        let tensor = creation::ones(&[3, 32, 32]).unwrap();
        let result = psnr(&tensor, &tensor, Some(1.0)).unwrap();
        assert!(result.is_infinite());
    }

    #[test]
    fn test_ssim_identical_images() {
        let tensor = creation::ones(&[3, 32, 32]).unwrap();
        let result = ssim(&tensor, &tensor, None, None, None).unwrap();
        assert!((result - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_invalid_tensor_shapes() {
        let tensor_2d = creation::ones(&[32, 32]).unwrap();
        let tensor_3d = creation::ones(&[3, 32, 32]).unwrap();

        assert!(mse(&tensor_2d, &tensor_3d).is_err());
        assert!(mae(&tensor_2d, &tensor_3d).is_err());
        assert!(psnr(&tensor_2d, &tensor_3d, None).is_err());
        assert!(ssim(&tensor_2d, &tensor_3d, None, None, None).is_err());
    }
}
