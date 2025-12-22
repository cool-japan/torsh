//! Basic image processing operations for vision workflows
//!
//! This module contains fundamental image processing functions including resizing,
//! normalization, and basic tensor manipulations for computer vision tasks.

use crate::{Result, VisionError};
use image::DynamicImage;
use torsh_tensor::{creation, Tensor};

/// Resize image using the image crate's built-in resizing
///
/// # Arguments
/// * `image` - Source image to resize
/// * `width` - Target width in pixels
/// * `height` - Target height in pixels
/// * `filter` - Resampling filter to use for resizing
///
/// # Returns
/// Resized image
///
/// # Example
/// ```
/// use torsh_vision::utils::image_processing::resize_image;
/// use image::imageops::FilterType;
///
/// let resized = resize_image(&image, 224, 224, FilterType::Lanczos3);
/// ```
pub fn resize_image(
    image: &DynamicImage,
    width: u32,
    height: u32,
    filter: image::imageops::FilterType,
) -> DynamicImage {
    image.resize(width, height, filter)
}

/// Normalize tensor pixel values to [0, 1] range
///
/// # Arguments
/// * `tensor` - Input tensor with pixel values (typically in [0, 255] range)
///
/// # Returns
/// Normalized tensor with values in [0, 1] range
pub fn normalize_tensor(tensor: &Tensor<f32>) -> Result<Tensor<f32>> {
    let mut normalized = tensor.clone();
    normalized.div_scalar_(255.0)?;
    Ok(normalized)
}

/// Denormalize tensor from [0, 1] to [0, 255] range
///
/// # Arguments
/// * `tensor` - Normalized tensor with values in [0, 1] range
///
/// # Returns
/// Denormalized tensor with values in [0, 255] range
pub fn denormalize_tensor(tensor: &Tensor<f32>) -> Result<Tensor<f32>> {
    let mut denormalized = tensor.clone();
    denormalized.mul_scalar_(255.0)?;
    Ok(denormalized)
}

/// Validate tensor shape for image processing operations
///
/// # Arguments
/// * `tensor` - Tensor to validate
/// * `expected_dims` - Expected number of dimensions (typically 3 for C×H×W)
///
/// # Returns
/// Ok(()) if valid, error otherwise
pub fn validate_image_tensor_shape(tensor: &Tensor<f32>, expected_dims: usize) -> Result<()> {
    let shape = tensor.shape();
    if shape.dims().len() != expected_dims {
        return Err(VisionError::InvalidShape(format!(
            "Expected {}D tensor, got {}D",
            expected_dims,
            shape.dims().len()
        )));
    }
    Ok(())
}

/// Clamp tensor values to specified range
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `min_val` - Minimum value
/// * `max_val` - Maximum value
///
/// # Returns
/// Tensor with clamped values
pub fn clamp_tensor(tensor: &Tensor<f32>, min_val: f32, max_val: f32) -> Result<Tensor<f32>> {
    let clamped = tensor.clamp(min_val, max_val)?;
    Ok(clamped)
}

/// Convert RGB tensor to grayscale using luminance weights
///
/// # Arguments
/// * `rgb_tensor` - RGB tensor with shape [3, H, W]
///
/// # Returns
/// Grayscale tensor with shape [1, H, W]
pub fn rgb_to_grayscale(rgb_tensor: &Tensor<f32>) -> Result<Tensor<f32>> {
    validate_image_tensor_shape(rgb_tensor, 3)?;

    let shape = rgb_tensor.shape();
    if shape.dims()[0] != 3 {
        return Err(VisionError::InvalidShape(
            "Expected RGB tensor with 3 channels".to_string(),
        ));
    }

    let height = shape.dims()[1];
    let width = shape.dims()[2];
    let grayscale = creation::zeros(&[1, height, width]).unwrap();

    // ITU-R BT.709 luma coefficients
    let r_weight = 0.2126;
    let g_weight = 0.7152;
    let b_weight = 0.0722;

    for h in 0..height {
        for w in 0..width {
            let r = rgb_tensor.get(&[0, h, w])?;
            let g = rgb_tensor.get(&[1, h, w])?;
            let b = rgb_tensor.get(&[2, h, w])?;

            let gray_val = r * r_weight + g * g_weight + b * b_weight;
            grayscale.set(&[0, h, w], gray_val)?;
        }
    }

    Ok(grayscale)
}
