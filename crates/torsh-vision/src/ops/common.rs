//! Common utilities and types for computer vision operations
//!
//! This module provides shared functionality used across different vision operation
//! implementations including error types, utility functions, and common patterns.

use crate::{Result, VisionError};
use torsh_tensor::Tensor;

/// Common validation and utility functions for vision operations
pub mod utils {
    use super::*;

    /// Validate that a tensor has the expected 3D shape (C, H, W)
    pub fn validate_image_tensor_3d(tensor: &Tensor<f32>) -> Result<(usize, usize, usize)> {
        let shape = tensor.shape();
        let dims = shape.dims();

        if dims.len() != 3 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (C, H, W), got {}D",
                dims.len()
            )));
        }

        Ok((dims[0], dims[1], dims[2]))
    }

    /// Validate that a tensor has the expected 4D shape (N, C, H, W)
    pub fn validate_image_tensor_4d(tensor: &Tensor<f32>) -> Result<(usize, usize, usize, usize)> {
        let shape = tensor.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 4D tensor (N, C, H, W), got {}D",
                dims.len()
            )));
        }

        Ok((dims[0], dims[1], dims[2], dims[3]))
    }

    /// Validate tensor for image operations, supporting both 3D (C, H, W) and 4D (N, C, H, W) formats
    /// Returns (batch_size, channels, height, width). For 3D tensors, batch_size=1.
    pub fn validate_image_tensor_flexible(
        tensor: &Tensor<f32>,
    ) -> Result<(usize, usize, usize, usize)> {
        let shape = tensor.shape();
        let dims = shape.dims();

        match dims.len() {
            3 => {
                // 3D tensor (C, H, W) - treat as single image with batch_size=1
                Ok((1, dims[0], dims[1], dims[2]))
            }
            4 => {
                // 4D tensor (N, C, H, W) - batched images
                Ok((dims[0], dims[1], dims[2], dims[3]))
            }
            _ => Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (C, H, W) or 4D tensor (N, C, H, W), got {}D",
                dims.len()
            ))),
        }
    }

    /// Validate that crop size is valid for the given image dimensions
    pub fn validate_crop_size(
        image_width: usize,
        image_height: usize,
        crop_width: usize,
        crop_height: usize,
    ) -> Result<()> {
        if crop_width > image_width || crop_height > image_height {
            return Err(VisionError::InvalidArgument(format!(
                "Crop size ({}, {}) larger than image size ({}, {})",
                crop_width, crop_height, image_width, image_height
            )));
        }
        Ok(())
    }

    /// Calculate bilinear interpolation weights for resizing
    pub fn bilinear_interpolation(
        src_x: f32,
        src_y: f32,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    ) -> (f32, f32, f32, f32) {
        let dx = src_x - x1 as f32;
        let dy = src_y - y1 as f32;

        let w11 = (1.0 - dx) * (1.0 - dy);
        let w21 = dx * (1.0 - dy);
        let w12 = (1.0 - dx) * dy;
        let w22 = dx * dy;

        (w11, w21, w12, w22)
    }

    /// Calculate the intersection over union (IoU) between two bounding boxes
    pub fn calculate_box_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
        let x1_max = box1[0].max(box2[0]);
        let y1_max = box1[1].max(box2[1]);
        let x2_min = box1[2].min(box2[2]);
        let y2_min = box1[3].min(box2[3]);

        if x2_min <= x1_max || y2_min <= y1_max {
            return 0.0;
        }

        let intersection = (x2_min - x1_max) * (y2_min - y1_max);
        let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
        let union = area1 + area2 - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    /// Clamp a value to be within bounds for image coordinates
    pub fn clamp_coord(value: i64, min_val: usize, max_val: usize) -> usize {
        if value < 0 {
            0
        } else if value >= max_val as i64 {
            max_val - 1
        } else {
            value as usize
        }
    }

    /// Calculate Gaussian kernel weights for a given sigma
    pub fn gaussian_kernel_1d(sigma: f32, kernel_size: usize) -> Vec<f32> {
        let mut kernel = vec![0.0; kernel_size];
        let center = kernel_size / 2;
        let sigma_sq = sigma * sigma;
        let pi_2_sigma_sq = 2.0 * std::f32::consts::PI * sigma_sq;

        let mut sum = 0.0;
        for i in 0..kernel_size {
            let distance = (i as i32 - center as i32).abs() as f32;
            let weight = (-distance * distance / (2.0 * sigma_sq)).exp() / pi_2_sigma_sq.sqrt();
            kernel[i] = weight;
            sum += weight;
        }

        // Normalize the kernel
        for weight in &mut kernel {
            *weight /= sum;
        }

        kernel
    }

    /// Create a morphological structuring element
    pub fn create_structuring_element(kernel_size: usize) -> Vec<Vec<bool>> {
        let mut kernel = vec![vec![false; kernel_size]; kernel_size];
        let center = kernel_size / 2;

        // Create a circular structuring element
        for y in 0..kernel_size {
            for x in 0..kernel_size {
                let dy = (y as i32 - center as i32).abs() as f32;
                let dx = (x as i32 - center as i32).abs() as f32;
                let distance = (dx * dx + dy * dy).sqrt();

                if distance <= center as f32 {
                    kernel[y][x] = true;
                }
            }
        }

        kernel
    }
}

/// Common constants used across vision operations
pub mod constants {
    /// Standard ImageNet normalization means for RGB channels
    pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

    /// Standard ImageNet normalization standard deviations for RGB channels
    pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    /// Weights for RGB to grayscale conversion (luminance formula)
    pub const RGB_TO_GRAY_WEIGHTS: [f32; 3] = [0.299, 0.587, 0.114];

    /// Sobel X kernel for edge detection
    pub const SOBEL_X_KERNEL: [[f32; 3]; 3] =
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

    /// Sobel Y kernel for edge detection
    pub const SOBEL_Y_KERNEL: [[f32; 3]; 3] =
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    /// Default Gaussian blur sigma
    pub const DEFAULT_GAUSSIAN_SIGMA: f32 = 1.0;

    /// Default Canny edge detection thresholds
    pub const DEFAULT_CANNY_LOW_THRESHOLD: f32 = 50.0;
    pub const DEFAULT_CANNY_HIGH_THRESHOLD: f32 = 150.0;
}

/// Interpolation methods for resizing operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Nearest neighbor interpolation (fastest, lowest quality)
    Nearest,
    /// Bilinear interpolation (good balance of speed and quality)
    Bilinear,
    /// Bicubic interpolation (slower, higher quality)
    Bicubic,
}

impl Default for InterpolationMode {
    fn default() -> Self {
        InterpolationMode::Bilinear
    }
}

/// Padding modes for convolution-like operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    /// Zero padding
    Zero,
    /// Reflect padding (mirror)
    Reflect,
    /// Replicate padding (extend edge values)
    Replicate,
    /// Circular/wrap padding
    Circular,
}

impl Default for PaddingMode {
    fn default() -> Self {
        PaddingMode::Zero
    }
}

/// Edge detection algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDetectionAlgorithm {
    /// Sobel edge detection
    Sobel,
    /// Canny edge detection
    Canny,
    /// Laplacian edge detection
    Laplacian,
    /// Prewitt edge detection
    Prewitt,
}

impl Default for EdgeDetectionAlgorithm {
    fn default() -> Self {
        EdgeDetectionAlgorithm::Sobel
    }
}

/// Morphological operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphologicalOperation {
    /// Erosion operation
    Erosion,
    /// Dilation operation
    Dilation,
    /// Opening operation (erosion followed by dilation)
    Opening,
    /// Closing operation (dilation followed by erosion)
    Closing,
    /// Gradient operation (dilation - erosion)
    Gradient,
    /// Top hat operation (opening - original)
    TopHat,
    /// Black hat operation (original - closing)
    BlackHat,
}

/// Configuration for various vision operations
#[derive(Debug, Clone)]
pub struct VisionOpConfig {
    /// Interpolation mode for resizing
    pub interpolation: InterpolationMode,
    /// Padding mode for filtering operations
    pub padding: PaddingMode,
    /// Whether to preserve aspect ratio in resize operations
    pub preserve_aspect_ratio: bool,
    /// Anti-aliasing for resize operations
    pub antialias: bool,
}

impl Default for VisionOpConfig {
    fn default() -> Self {
        Self {
            interpolation: InterpolationMode::default(),
            padding: PaddingMode::default(),
            preserve_aspect_ratio: false,
            antialias: true,
        }
    }
}

impl VisionOpConfig {
    /// Create configuration optimized for quality
    pub fn high_quality() -> Self {
        Self {
            interpolation: InterpolationMode::Bicubic,
            antialias: true,
            ..Default::default()
        }
    }

    /// Create configuration optimized for speed
    pub fn fast() -> Self {
        Self {
            interpolation: InterpolationMode::Nearest,
            antialias: false,
            ..Default::default()
        }
    }

    /// Create configuration for preserving aspect ratio
    pub fn preserve_aspect() -> Self {
        Self {
            preserve_aspect_ratio: true,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_validate_image_tensor_3d() {
        let tensor = zeros(&[3, 224, 224]).unwrap();
        let (c, h, w) = utils::validate_image_tensor_3d(&tensor).unwrap();
        assert_eq!((c, h, w), (3, 224, 224));

        let invalid_tensor = zeros(&[224, 224]).unwrap();
        assert!(utils::validate_image_tensor_3d(&invalid_tensor).is_err());
    }

    #[test]
    fn test_validate_crop_size() {
        assert!(utils::validate_crop_size(224, 224, 128, 128).is_ok());
        assert!(utils::validate_crop_size(224, 224, 256, 128).is_err());
        assert!(utils::validate_crop_size(224, 224, 128, 256).is_err());
    }

    #[test]
    fn test_bilinear_interpolation_weights() {
        let (w11, w21, w12, w22) = utils::bilinear_interpolation(1.5, 1.5, 1, 1, 2, 2);
        assert!((w11 + w21 + w12 + w22 - 1.0).abs() < 1e-6);
        assert_eq!(w11, 0.25);
        assert_eq!(w21, 0.25);
        assert_eq!(w12, 0.25);
        assert_eq!(w22, 0.25);
    }

    #[test]
    fn test_box_iou_calculation() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [5.0, 5.0, 15.0, 15.0];
        let iou = utils::calculate_box_iou(&box1, &box2);
        assert!((iou - 0.142857).abs() < 1e-5); // 25 / 175 ≈ 0.142857

        let identical_box = [0.0, 0.0, 10.0, 10.0];
        let iou_identical = utils::calculate_box_iou(&box1, &identical_box);
        assert!((iou_identical - 1.0).abs() < 1e-6);

        let non_overlapping = [20.0, 20.0, 30.0, 30.0];
        let iou_zero = utils::calculate_box_iou(&box1, &non_overlapping);
        assert_eq!(iou_zero, 0.0);
    }

    #[test]
    fn test_gaussian_kernel() {
        let kernel = utils::gaussian_kernel_1d(1.0, 5);
        assert_eq!(kernel.len(), 5);

        // Sum should be approximately 1
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Center should have highest value
        assert!(kernel[2] > kernel[1]);
        assert!(kernel[2] > kernel[3]);
    }

    #[test]
    fn test_config_presets() {
        let high_quality = VisionOpConfig::high_quality();
        assert_eq!(high_quality.interpolation, InterpolationMode::Bicubic);
        assert!(high_quality.antialias);

        let fast = VisionOpConfig::fast();
        assert_eq!(fast.interpolation, InterpolationMode::Nearest);
        assert!(!fast.antialias);

        let preserve_aspect = VisionOpConfig::preserve_aspect();
        assert!(preserve_aspect.preserve_aspect_ratio);
    }
}

/// Convert image::DynamicImage to Tensor<f32>
///
/// This is a convenience function that wraps the image_to_tensor utility function
/// from the utils module.
pub fn to_tensor(image: &image::DynamicImage) -> Result<Tensor<f32>> {
    crate::utils::image_to_tensor(image)
}
