//! Image and tensor conversion utilities
//!
//! This module provides core conversion functions between tensors and images,
//! enabling seamless integration between computer vision models and image processing pipelines.

use crate::{Result, VisionError};
use image::{DynamicImage, GenericImageView};
use torsh_tensor::{creation, Tensor};

/// Convert tensor to image
///
/// Converts a 3D tensor with shape (C, H, W) to a DynamicImage.
/// Supports RGB (3 channels) and grayscale (1 channel) images.
///
/// # Arguments
/// * `tensor` - Input tensor with shape (C, H, W) where C is 1 or 3
/// * `normalize` - Whether to normalize pixel values to [0, 1] range
///
/// # Returns
/// A `DynamicImage` that can be saved or further processed
///
/// # Example
/// ```rust
/// use torsh_tensor::creation;
/// use torsh_vision::utils::image_conversion::tensor_to_image;
///
/// // Create a random RGB tensor
/// let tensor = creation::randn(&[3, 224, 224]).unwrap();
/// let image = tensor_to_image(&tensor, true)?;
/// ```
///
/// # Errors
/// Returns `VisionError::InvalidShape` if tensor is not 3D or doesn't have 1 or 3 channels
pub fn tensor_to_image(tensor: &Tensor<f32>, normalize: bool) -> Result<DynamicImage> {
    let shape = tensor.shape();
    if shape.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensor (C, H, W), got {}D",
            shape.dims().len()
        )));
    }

    let (channels, height, width) = (shape.dims()[0], shape.dims()[1], shape.dims()[2]);

    if channels != 3 && channels != 1 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 1 or 3 channels, got {}",
            channels
        )));
    }

    let mut processed_tensor = tensor.clone();

    // Normalize to [0, 1] if requested
    if normalize {
        let min_val = processed_tensor.min()?;
        let max_val = processed_tensor.max(None, false)?;
        let min_val_f32 = min_val.to_vec()?[0];
        let max_val_f32 = max_val.to_vec()?[0];

        if max_val_f32 > min_val_f32 {
            processed_tensor.sub_scalar_(min_val_f32)?;
            processed_tensor = processed_tensor.div_scalar(max_val_f32 - min_val_f32)?;
        }
    }

    if channels == 3 {
        // RGB image
        let mut img_buffer = image::RgbImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let r = (processed_tensor.get(&[0, y, x])? * 255.0).clamp(0.0, 255.0) as u8;
                let g = (processed_tensor.get(&[1, y, x])? * 255.0).clamp(0.0, 255.0) as u8;
                let b = (processed_tensor.get(&[2, y, x])? * 255.0).clamp(0.0, 255.0) as u8;

                img_buffer.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            }
        }

        Ok(DynamicImage::ImageRgb8(img_buffer))
    } else {
        // Grayscale image
        let mut img_buffer = image::GrayImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let gray = (processed_tensor.get(&[0, y, x])? * 255.0).clamp(0.0, 255.0) as u8;
                img_buffer.put_pixel(x as u32, y as u32, image::Luma([gray]));
            }
        }

        Ok(DynamicImage::ImageLuma8(img_buffer))
    }
}

/// Convert image to tensor
///
/// Converts a DynamicImage to a 3D tensor with shape (C, H, W).
/// Automatically handles different image formats (RGB, RGBA, grayscale) and
/// converts pixel values to the range [0, 1].
///
/// # Arguments
/// * `image` - Input image to convert
///
/// # Returns
/// A tensor with shape (C, H, W) where:
/// - C = 3 for RGB images (or converted from other formats)
/// - C = 1 for grayscale images
/// - Values are normalized to [0, 1] range
///
/// # Example
/// ```rust
/// use image::DynamicImage;
/// use torsh_vision::utils::image_conversion::image_to_tensor;
///
/// let image = image::open("example.jpg")?;
/// let tensor = image_to_tensor(&image)?;
/// println!("Tensor shape: {:?}", tensor.shape().dims());
/// ```
///
/// # Note
/// RGBA and other formats are automatically converted to RGB.
/// Pixel values are normalized from [0, 255] to [0, 1] range.
pub fn image_to_tensor(image: &DynamicImage) -> Result<Tensor<f32>> {
    let (width, height) = image.dimensions();

    match image {
        DynamicImage::ImageRgb8(rgb_img) => {
            let mut tensor = creation::zeros(&[3, height as usize, width as usize]).unwrap();

            for y in 0..height {
                for x in 0..width {
                    let pixel = rgb_img.get_pixel(x, y);
                    let r = pixel[0] as f32 / 255.0;
                    let g = pixel[1] as f32 / 255.0;
                    let b = pixel[2] as f32 / 255.0;

                    tensor.set(&[0, y as usize, x as usize], r)?;
                    tensor.set(&[1, y as usize, x as usize], g)?;
                    tensor.set(&[2, y as usize, x as usize], b)?;
                }
            }

            Ok(tensor)
        }
        DynamicImage::ImageLuma8(gray_img) => {
            let mut tensor = creation::zeros(&[1, height as usize, width as usize]).unwrap();

            for y in 0..height {
                for x in 0..width {
                    let pixel = gray_img.get_pixel(x, y);
                    let gray = pixel[0] as f32 / 255.0;

                    tensor.set(&[0, y as usize, x as usize], gray)?;
                }
            }

            Ok(tensor)
        }
        _ => {
            // Convert to RGB first
            let rgb_image = image.to_rgb8();
            let mut tensor = creation::zeros(&[3, height as usize, width as usize]).unwrap();

            for y in 0..height {
                for x in 0..width {
                    let pixel = rgb_image.get_pixel(x, y);
                    let r = pixel[0] as f32 / 255.0;
                    let g = pixel[1] as f32 / 255.0;
                    let b = pixel[2] as f32 / 255.0;

                    tensor.set(&[0, y as usize, x as usize], r)?;
                    tensor.set(&[1, y as usize, x as usize], g)?;
                    tensor.set(&[2, y as usize, x as usize], b)?;
                }
            }

            Ok(tensor)
        }
    }
}

/// Denormalize tensor (reverse of normalize operation)
///
/// Applies denormalization to a tensor using per-channel mean and standard deviation.
/// This is commonly used to reverse normalization applied during preprocessing,
/// particularly useful for visualizing model outputs or intermediate features.
///
/// # Arguments
/// * `tensor` - Input tensor with shape (C, H, W)
/// * `mean` - Per-channel mean values for denormalization
/// * `std` - Per-channel standard deviation values for denormalization
///
/// # Returns
/// Denormalized tensor with same shape as input
///
/// # Formula
/// For each pixel value x: `denormalized_x = x * std + mean`
///
/// # Example
/// ```rust
/// use torsh_tensor::creation;
/// use torsh_vision::utils::image_conversion::denormalize;
///
/// let tensor = creation::randn(&[3, 224, 224]).unwrap();
/// let mean = [0.485, 0.456, 0.406];  // ImageNet means
/// let std = [0.229, 0.224, 0.225];   // ImageNet stds
/// let denormalized = denormalize(&tensor, &mean, &std)?;
/// ```
///
/// # Errors
/// Returns error if:
/// - Tensor is not 3D
/// - Length of mean/std arrays doesn't match number of channels
/// - Any standard deviation value is zero
pub fn denormalize(tensor: &Tensor<f32>, mean: &[f32], std: &[f32]) -> Result<Tensor<f32>> {
    let shape = tensor.shape();
    if shape.dims().len() != 3 {
        return Err(VisionError::InvalidShape(format!(
            "Expected 3D tensor (C, H, W), got {}D",
            shape.dims().len()
        )));
    }

    let channels = shape.dims()[0];
    if mean.len() != channels || std.len() != channels {
        return Err(VisionError::InvalidArgument(format!(
            "Mean and std must have same length as number of channels. Got {} channels, {} mean values, {} std values",
            channels, mean.len(), std.len()
        )));
    }

    let mut output = tensor.clone();

    for c in 0..channels {
        let channel_mean = mean[c];
        let channel_std = std[c];

        if channel_std == 0.0 {
            return Err(VisionError::InvalidArgument(
                "Standard deviation cannot be zero".to_string(),
            ));
        }

        // Apply denormalization: x * std + mean
        for y in 0..shape.dims()[1] {
            for x in 0..shape.dims()[2] {
                let val = output.get(&[c, y, x])?;
                let denormalized_val = val * channel_std + channel_mean;
                output.set(&[c, y, x], denormalized_val)?;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_tensor_to_image_rgb() {
        let tensor = creation::ones(&[3, 10, 10]).unwrap();
        let image = tensor_to_image(&tensor, false).unwrap();

        match image {
            DynamicImage::ImageRgb8(rgb_img) => {
                assert_eq!(rgb_img.width(), 10);
                assert_eq!(rgb_img.height(), 10);
            }
            _ => panic!("Expected RGB image"),
        }
    }

    #[test]
    fn test_tensor_to_image_grayscale() {
        let tensor = creation::ones(&[1, 10, 10]).unwrap();
        let image = tensor_to_image(&tensor, false).unwrap();

        match image {
            DynamicImage::ImageLuma8(gray_img) => {
                assert_eq!(gray_img.width(), 10);
                assert_eq!(gray_img.height(), 10);
            }
            _ => panic!("Expected grayscale image"),
        }
    }

    #[test]
    fn test_image_to_tensor_roundtrip() {
        let original_tensor = creation::rand(&[3, 5, 5]).unwrap();
        let image = tensor_to_image(&original_tensor, false).unwrap();
        let converted_tensor = image_to_tensor(&image).unwrap();

        assert_eq!(converted_tensor.shape().dims(), &[3, 5, 5]);
    }

    #[test]
    fn test_denormalize() {
        let tensor = creation::zeros(&[3, 2, 2]).unwrap();
        let mean = [0.5, 0.5, 0.5];
        let std = [0.2, 0.2, 0.2];

        let result = denormalize(&tensor, &mean, &std).unwrap();

        // All values should be equal to the mean since input was zeros
        for c in 0..3 {
            for y in 0..2 {
                for x in 0..2 {
                    let val = result.get(&[c, y, x]).unwrap();
                    assert!((val - 0.5).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_denormalize_invalid_channels() {
        let tensor = creation::zeros(&[3, 2, 2]).unwrap();
        let mean = [0.5, 0.5]; // Wrong length
        let std = [0.2, 0.2, 0.2];

        assert!(denormalize(&tensor, &mean, &std).is_err());
    }

    #[test]
    fn test_denormalize_zero_std() {
        let tensor = creation::zeros(&[3, 2, 2]).unwrap();
        let mean = [0.5, 0.5, 0.5];
        let std = [0.2, 0.0, 0.2]; // Zero std

        assert!(denormalize(&tensor, &mean, &std).is_err());
    }
}
