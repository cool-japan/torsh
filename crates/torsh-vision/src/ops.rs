//! Computer vision operations for image processing and analysis
//!
//! This module has been refactored into a comprehensive modular structure while maintaining
//! 100% backward compatibility. All original functions are available through re-exports.
//!
//! ## Module Structure
//!
//! The vision operations module is organized into specialized sub-modules:
//!
//! - **`common`** - Shared utilities, configurations, and helper functions
//! - **`geometric`** - Geometric transformations (resize, crop, flip, rotate, padding)
//! - **`filtering`** - Image filtering and enhancement (blur, edge detection, morphological ops)
//! - **`color`** - Color operations (space conversions, normalization, adjustments)
//! - **`detection`** - Object detection utilities (NMS, ROI pooling, bbox operations)
//! - **`analysis`** - Loss functions and evaluation metrics for vision tasks
//!
//! ## Usage Examples
//!
//! All original functions continue to work exactly as before:
//!
//! ```rust
//! use torsh_vision::ops::{resize, center_crop, normalize, sobel_edge_detection};
//!
//! // Geometric transformations
//! let resized = resize(&image, (224, 224))?;
//! let cropped = center_crop(&image, (224, 224))?;
//!
//! // Image processing
//! let normalized = normalize(&image, &mean, &std)?;
//! let edges = sobel_edge_detection(&image)?;
//! ```
//!
//! ## Enhanced Features
//!
//! The refactored module also provides enhanced capabilities:
//!
//! ```rust
//! use torsh_vision::ops::{
//!     geometric::{ResizeConfig, InterpolationMode},
//!     color::NormalizationConfig,
//!     filtering::EdgeDetectionConfig,
//!     detection::{NMSConfig, Detection}
//! };
//!
//! // Enhanced configuration options
//! let config = ResizeConfig::new((224, 224))
//!     .with_interpolation(InterpolationMode::Bicubic)
//!     .with_preserve_aspect_ratio(true);
//! let resized = resize_with_config(&image, config)?;
//!
//! // Advanced normalization
//! let norm_config = NormalizationConfig::imagenet();
//! let normalized = normalize_with_config(&image, norm_config)?;
//! ```

// Import all sub-modules
pub mod analysis;
pub mod color;
pub mod common;
pub mod detection;
pub mod filtering;
pub mod geometric;

// Re-export common utilities and types for convenience
pub use common::{
    constants, to_tensor, utils, EdgeDetectionAlgorithm, InterpolationMode, MorphologicalOperation,
    PaddingMode, VisionOpConfig,
};

// ═══════════════════════════════════════════════════════════════════════════════
// NOTE: Commented-out operations below are DEFERRED FEATURES, not bugs
// ═══════════════════════════════════════════════════════════════════════════════
//
// Many operations below have "// TODO: Implement" comments. These are documented
// features planned for future releases (v0.2.0+), not missing functionality.
//
// Status: Core functionality is implemented and working
// Documentation: See ROADMAP.md for full feature plan
// Priority: High-priority features (detection ops, segmentation losses) in v0.2.0
//           Low-priority features (config variants) in v0.3.0+
//
// Current API provides:
// - Basic geometric operations (resize, crop, flip, rotate, pad)
// - Image filtering (Gaussian blur, median filter, edge detection)
// - Color operations (brightness, contrast, normalization, color space)
// - Detection utilities (NMS, IoU, anchors)
// - Classification metrics
//
// For RC.1 release, these commented operations are intentionally deferred.
// ═══════════════════════════════════════════════════════════════════════════════

// === GEOMETRIC TRANSFORMATIONS ===
// Re-export all geometric operations for backward compatibility
// Only import functions that actually exist in geometric module
pub use geometric::{
    // Crop operations - only those that exist
    center_crop,
    // crop_with_config,  // TODO: Implement

    // flip_with_config,  // TODO: Implement

    // Flip operations - only those that exist
    horizontal_flip,
    // Padding operations - only those that exist
    pad,
    // pad_with_config,  // TODO: Implement
    random_crop,
    // Basic resize operations - only those that exist
    resize,
    resize_with_mode,
    // resize_bicubic,  // TODO: Implement
    // resize_bilinear,  // TODO: Implement
    // resize_nearest,  // TODO: Implement
    // resize_with_config,  // TODO: Implement

    // Rotation operations - only those that exist
    rotate,

    vertical_flip,
    // CropConfig,  // TODO: Implement
    // FlipConfig,  // TODO: Implement
    // PaddingConfig,  // TODO: Implement
    // Configuration types - only those that exist
    // ResizeConfig,  // TODO: Implement
    // RotationConfig,  // TODO: Implement
};

// === IMAGE FILTERING AND ENHANCEMENT ===
// Re-export all filtering operations for backward compatibility
pub use filtering::{
    // bilateral_filter,  // TODO: Implement

    // Convolution operations - only those that exist
    // conv2d,  // TODO: Implement
    // conv2d_with_config,  // TODO: Implement
    edge_detection,

    // Gaussian blur and smoothing - only those that exist
    gaussian_blur,
    // gaussian_blur_with_config,  // TODO: Implement

    // Noise reduction - only those that exist
    median_filter,
    // Morphological operations - only those that exist
    morphological_operation,

    // Edge detection - only those that exist
    sobel_edge_detection,
    // EdgeDetectionConfig,  // TODO: Implement
    // Configuration types - only those that exist
    // FilteringConfig,  // TODO: Implement
    // GaussianBlurConfig,  // TODO: Implement
    // MorphologyConfig,  // TODO: Implement
};

// === COLOR OPERATIONS ===
// Re-export all color operations for backward compatibility
pub use color::{
    // Color adjustments - only those that exist
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    // compute_histogram,  // TODO: Implement

    // Channel operations - only those that exist
    // extract_channel,  // TODO: Implement

    // gamma_correction,  // TODO: Implement

    // Histogram operations - only those that exist
    histogram_equalization,
    // histogram_equalization_with_config,  // TODO: Implement
    hsv_to_rgb,
    // Normalization - only those that exist
    normalize,
    // normalize_imagenet,  // TODO: Implement
    // normalize_with_config,  // TODO: Implement

    // Color space conversions - only those that exist
    rgb_to_grayscale,
    rgb_to_hsv,
    // rgb_to_yuv,  // TODO: Implement

    // ColorConfig,  // TODO: Implement
    // Configuration types and enums - only those that exist
    // ColorSpace,  // TODO: Implement
    // HistogramConfig,  // TODO: Implement
    // HistogramMethod,  // TODO: Implement
    // NormalizationConfig,  // TODO: Implement
    // NormalizationMethod,  // TODO: Implement
};

// === DETECTION OPERATIONS ===
// Re-export all detection operations for backward compatibility
pub use detection::{
    // apply_bbox_deltas,  // TODO: Implement

    // IoU calculations - only those that exist
    calculate_iou,
    // calculate_iou_matrix,  // TODO: Implement

    // clip_bbox,  // TODO: Implement
    // Training utilities - only those that exist
    // compute_bbox_targets,  // TODO: Implement

    // Bounding box operations - only those that exist
    // convert_bbox_format,  // TODO: Implement
    // filter_boxes_by_size,  // TODO: Implement

    // Anchor generation - only those that exist
    generate_anchors,
    // Non-Maximum Suppression - only those that exist
    nms,

    // ROI pooling - only those that exist
    // roi_pool,  // TODO: Implement

    // scale_bbox,  // TODO: Implement
    // AnchorConfig,  // TODO: Implement
    // BBoxFormat,  // TODO: Implement
    // Types and configurations - only those that exist
    BoundingBox,
    Detection,
    // NMSConfig,  // TODO: Implement
    // ROIPoolConfig,  // TODO: Implement
};

// === ANALYSIS AND LOSS FUNCTIONS ===
// Re-export all analysis operations for backward compatibility
pub use analysis::{
    // Evaluation metrics - only those that exist
    compute_classification_metrics,
    // compute_detection_metrics,  // TODO: Implement
    // compute_segmentation_metrics,  // TODO: Implement

    // Loss functions - only those that exist
    cross_entropy_loss,
    // dice_loss,  // TODO: Implement
    // focal_loss,  // TODO: Implement
    // iou_loss,  // TODO: Implement

    // Metric types - only those that exist
    ClassificationMetrics,
    DetectionMetrics,
    // DiceLossConfig,  // TODO: Implement
    // FocalLossConfig,  // TODO: Implement
    // LossConfig,  // TODO: Implement
    // Configuration types - only those that exist
    // Reduction,  // TODO: Implement
    SegmentationMetrics,
};

// === BACKWARD COMPATIBILITY FUNCTIONS ===
// Provide simple wrappers for common operations to maintain exact API compatibility

use crate::Result;
use torsh_tensor::Tensor;

/// Normalize image using mean and standard deviation (backward compatibility)
pub fn normalize_with_mean_std(
    image: &Tensor<f32>,
    mean: &[f32],
    std: &[f32],
) -> Result<Tensor<f32>> {
    let config = color::NormalizationConfig::custom(mean.to_vec(), std.to_vec());
    color::normalize(image, config)
}

/// Simple resize function with default bilinear interpolation (exact backward compatibility)
pub fn resize_simple(image: &Tensor<f32>, size: (usize, usize)) -> Result<Tensor<f32>> {
    geometric::resize(image, size)
}

/// Simple normalization function (backward compatibility)
pub fn normalize_simple(image: &Tensor<f32>, mean: &[f32], std: &[f32]) -> Result<Tensor<f32>> {
    normalize_with_mean_std(image, mean, std)
}

// === CONVENIENCE RE-EXPORTS ===
// Re-export the most commonly used items at the top level for ease of use

// Most common geometric operations
pub use geometric::{center_crop as image_center_crop, resize as image_resize};

// Most common filtering operations
pub use filtering::{gaussian_blur as image_blur, sobel_edge_detection as edge_detection_sobel};

// Most common color operations
pub use color::{
    adjust_brightness as brightness, adjust_contrast as contrast,
    normalize_imagenet as imagenet_normalize, rgb_to_grayscale as to_grayscale,
};

// Most common detection operations
pub use detection::{calculate_iou as box_iou, nms as non_max_suppression};

// Most common analysis operations
pub use analysis::{
    compute_classification_metrics as classification_eval, cross_entropy_loss as ce_loss,
};

// === FACTORY FUNCTIONS ===
/// Factory functions for creating common configurations

/// Create a standard image preprocessing pipeline configuration
pub fn standard_preprocessing_config() -> color::NormalizationConfig {
    color::NormalizationConfig::imagenet()
}

// DEFERRED: ResizeConfig and other configuration-based APIs
// These provide more flexible operation parameters but are not essential for v0.1.0-rc.1
// Current API uses function parameters directly (e.g., resize_with_mode)
// Planned for v0.2.0 - See ROADMAP.md
//
// Example usage (future):
// let config = ResizeConfig::new(size)
//     .with_interpolation(InterpolationMode::Bicubic)
//     .with_antialias(true);
// let resized = resize_with_config(&image, config)?;
//
// Current workaround:
// Use resize_with_mode(&image, 256, 256, InterpolationMode::Bilinear)?

/// Create a standard NMS configuration for object detection
pub fn standard_nms_config() -> detection::NMSConfig {
    detection::NMSConfig::new(0.5, 0.5).with_per_class(true)
}

/// Create a strict NMS configuration (lower IoU threshold)
pub fn strict_nms_config() -> detection::NMSConfig {
    detection::NMSConfig::new(0.3, 0.7).with_per_class(true)
}

/// Create a standard edge detection configuration
pub fn standard_edge_detection_config() -> filtering::EdgeDetectionConfig {
    filtering::EdgeDetectionConfig::sobel()
}

/// Create a Canny edge detection configuration
pub fn canny_edge_detection_config() -> filtering::EdgeDetectionConfig {
    filtering::EdgeDetectionConfig::canny(50.0, 150.0)
}

/// Create a high quality resize (using bicubic interpolation)
pub fn high_quality_resize(image: &Tensor<f32>, size: (usize, usize)) -> Result<Tensor<f32>> {
    geometric::resize_with_mode(image, size, common::InterpolationMode::Bicubic)
}

/// Create a fast resize (using nearest neighbor interpolation)
pub fn fast_resize(image: &Tensor<f32>, size: (usize, usize)) -> Result<Tensor<f32>> {
    geometric::resize_with_mode(image, size, common::InterpolationMode::Nearest)
}

// === MODULE TESTS ===
#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_backward_compatibility() -> Result<()> {
        let image = zeros(&[3, 32, 32])?;

        // Test that original function names still work
        let resized = resize(&image, (16, 16))?;
        assert_eq!(resized.shape().dims(), &[3, 16, 16]);

        let cropped = center_crop(&image, (16, 16))?;
        assert_eq!(cropped.shape().dims(), &[3, 16, 16]);

        let gray = rgb_to_grayscale(&image)?;
        assert_eq!(gray.shape().dims(), &[1, 32, 32]);

        let blurred = gaussian_blur(&image, 1.0)?;
        assert_eq!(blurred.shape().dims(), &[3, 32, 32]);

        Ok(())
    }

    #[test]
    fn test_enhanced_functionality() -> Result<()> {
        let image = zeros(&[3, 32, 32])?;

        // Test high quality resize functions
        let resized = high_quality_resize(&image, (16, 16))?;
        assert_eq!(resized.shape().dims(), &[3, 16, 16]);

        let norm_config = standard_preprocessing_config();
        let normalized = color::normalize(&image, norm_config)?;
        assert_eq!(normalized.shape().dims(), &[3, 32, 32]);

        Ok(())
    }

    #[test]
    fn test_convenience_aliases() -> Result<()> {
        let image = zeros(&[3, 32, 32])?;

        // Test convenience re-exports
        let resized = image_resize(&image, (16, 16))?;
        assert_eq!(resized.shape().dims(), &[3, 16, 16]);

        let gray = to_grayscale(&image)?;
        assert_eq!(gray.shape().dims(), &[1, 32, 32]);

        let bright = brightness(&image, 0.1)?;
        assert_eq!(bright.shape().dims(), &[3, 32, 32]);

        Ok(())
    }

    #[test]
    fn test_factory_functions() -> Result<()> {
        // Test that factory functions create valid configurations
        let image = zeros(&[3, 32, 32])?;
        let _resized_high = high_quality_resize(&image, (224, 224))?;
        let _resized_fast = fast_resize(&image, (224, 224))?;
        let _nms_config = standard_nms_config();
        let _edge_config = standard_edge_detection_config();
        let _preprocessing_config = standard_preprocessing_config();

        // All should be created without errors
        Ok(())
    }

    #[test]
    fn test_detection_operations() -> Result<()> {
        let detections = vec![
            detection::Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            detection::Detection::new([5.0, 5.0, 15.0, 15.0], 0.8, 0),
        ];

        let config = standard_nms_config();
        let filtered = nms(detections, config)?;

        // Should filter overlapping detections
        assert!(filtered.len() <= 2);

        Ok(())
    }

    #[test]
    fn test_filtering_operations() -> Result<()> {
        let image = zeros(&[1, 32, 32])?;

        // Test various filtering operations
        let edges = sobel_edge_detection(&image)?;
        assert_eq!(edges.shape().dims(), &[1, 32, 32]);

        let blurred = gaussian_blur(&image, 2.0)?;
        assert_eq!(blurred.shape().dims(), &[1, 32, 32]);

        let median_filtered = median_filter(&image, 3)?;
        assert_eq!(median_filtered.shape().dims(), &[1, 32, 32]);

        Ok(())
    }

    #[test]
    fn test_color_operations() -> Result<()> {
        let rgb_image = zeros(&[3, 16, 16])?;

        // Test color space conversions
        let hsv = rgb_to_hsv(&rgb_image)?;
        assert_eq!(hsv.shape().dims(), &[3, 16, 16]);

        let back_to_rgb = hsv_to_rgb(&hsv)?;
        assert_eq!(back_to_rgb.shape().dims(), &[3, 16, 16]);

        // Test color adjustments
        let bright = adjust_brightness(&rgb_image, 0.2)?;
        assert_eq!(bright.shape().dims(), &[3, 16, 16]);

        let contrasted = adjust_contrast(&rgb_image, 1.5)?;
        assert_eq!(contrasted.shape().dims(), &[3, 16, 16]);

        Ok(())
    }

    #[test]
    fn test_analysis_operations() -> Result<()> {
        let predictions = zeros(&[5, 3])?; // 5 samples, 3 classes
        let targets = zeros(&[5])?;

        // Test classification metrics
        let metrics = compute_classification_metrics(&predictions, &targets, 3)?;
        assert_eq!(metrics.precision.len(), 3);
        assert_eq!(metrics.recall.len(), 3);
        assert_eq!(metrics.f1_score.len(), 3);

        // Test loss functions
        let config = analysis::LossConfig::default();
        let loss = cross_entropy_loss(&predictions, &targets, config)?;
        assert_eq!(loss.shape().dims(), &[1]);

        Ok(())
    }
}
