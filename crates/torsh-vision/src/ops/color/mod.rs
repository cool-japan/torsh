//! Color operations and transformations for computer vision
//!
//! This module provides comprehensive color space conversions, normalization operations,
//! and color adjustment functions including:
//! - Color space conversions (RGB ↔ Grayscale, HSV, LAB, YUV)
//! - Image normalization with statistical methods
//! - Histogram operations and equalization
//! - Color adjustments (brightness, contrast, saturation, hue)
//! - Channel-wise operations and manipulations

use crate::ops::common::{constants, utils};
use crate::{Result, VisionError};
use torsh_tensor::creation::{full, ones, zeros, zeros_mut};
use torsh_tensor::Tensor;

/// Color space enumeration for conversions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// RGB color space (Red, Green, Blue)
    RGB,
    /// Grayscale (single channel)
    Grayscale,
    /// HSV color space (Hue, Saturation, Value)
    HSV,
    /// HSL color space (Hue, Saturation, Lightness)
    HSL,
    /// LAB color space (L*a*b*)
    LAB,
    /// YUV color space (Luminance, U chrominance, V chrominance)
    YUV,
    /// YCbCr color space (Y luma, Cb chroma blue, Cr chroma red)
    YCbCr,
}

/// Normalization method for image preprocessing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Min-max normalization to [0, 1] range
    MinMax,
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// ImageNet standard normalization
    ImageNet,
    /// Custom normalization with provided mean and std
    Custom,
}

/// Histogram equalization method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramMethod {
    /// Global histogram equalization
    Global,
    /// Adaptive histogram equalization (CLAHE)
    Adaptive,
    /// Local histogram equalization
    Local,
}

/// Configuration for color operations
#[derive(Debug, Clone)]
pub struct ColorConfig {
    /// Target color space for conversions
    pub target_space: ColorSpace,
    /// Whether to preserve original data type
    pub preserve_dtype: bool,
    /// Clamping range for output values
    pub clamp_range: Option<(f32, f32)>,
}

impl Default for ColorConfig {
    fn default() -> Self {
        Self {
            target_space: ColorSpace::RGB,
            preserve_dtype: true,
            clamp_range: Some((0.0, 1.0)),
        }
    }
}

impl ColorConfig {
    /// Create config for RGB to grayscale conversion
    pub fn rgb_to_grayscale() -> Self {
        Self {
            target_space: ColorSpace::Grayscale,
            ..Default::default()
        }
    }

    /// Create config for HSV conversion
    pub fn hsv() -> Self {
        Self {
            target_space: ColorSpace::HSV,
            ..Default::default()
        }
    }

    /// Create config for LAB conversion
    pub fn lab() -> Self {
        Self {
            target_space: ColorSpace::LAB,
            clamp_range: Some((-100.0, 100.0)), // LAB has different ranges
            ..Default::default()
        }
    }
}

/// Configuration for normalization operations
#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    /// Normalization method to use
    pub method: NormalizationMethod,
    /// Custom mean values (for custom normalization)
    pub mean: Option<Vec<f32>>,
    /// Custom standard deviation values (for custom normalization)
    pub std: Option<Vec<f32>>,
    /// Whether to apply per-channel normalization
    pub per_channel: bool,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            method: NormalizationMethod::MinMax,
            mean: None,
            std: None,
            per_channel: true,
            eps: 1e-8,
        }
    }
}

impl NormalizationConfig {
    /// Create ImageNet normalization configuration
    pub fn imagenet() -> Self {
        Self {
            method: NormalizationMethod::ImageNet,
            mean: Some(constants::IMAGENET_MEAN.to_vec()),
            std: Some(constants::IMAGENET_STD.to_vec()),
            per_channel: true,
            eps: 1e-8,
        }
    }

    /// Create custom normalization configuration
    pub fn custom(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self {
            method: NormalizationMethod::Custom,
            mean: Some(mean),
            std: Some(std),
            per_channel: true,
            eps: 1e-8,
        }
    }

    /// Create z-score normalization configuration
    pub fn zscore() -> Self {
        Self {
            method: NormalizationMethod::ZScore,
            per_channel: true,
            eps: 1e-8,
            ..Default::default()
        }
    }

    /// Set per-channel normalization
    pub fn with_per_channel(mut self, per_channel: bool) -> Self {
        self.per_channel = per_channel;
        self
    }
}

/// Configuration for histogram operations
#[derive(Debug, Clone)]
pub struct HistogramConfig {
    /// Histogram equalization method
    pub method: HistogramMethod,
    /// Number of bins for histogram computation
    pub bins: usize,
    /// Clip limit for adaptive histogram equalization
    pub clip_limit: f32,
    /// Tile size for adaptive methods
    pub tile_size: (usize, usize),
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            method: HistogramMethod::Global,
            bins: 256,
            clip_limit: 0.01,
            tile_size: (8, 8),
        }
    }
}

impl HistogramConfig {
    /// Create configuration for adaptive histogram equalization
    pub fn adaptive(clip_limit: f32, tile_size: (usize, usize)) -> Self {
        Self {
            method: HistogramMethod::Adaptive,
            clip_limit,
            tile_size,
            ..Default::default()
        }
    }
}

/// Convert RGB image to grayscale using luminance formula
pub fn rgb_to_grayscale(image: &Tensor<f32>) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if channels != 3 {
        return Err(VisionError::InvalidArgument(
            "RGB to grayscale conversion requires 3-channel input".to_string(),
        ));
    }

    let result = zeros_mut(&[1, height, width]);

    for y in 0..height {
        for x in 0..width {
            let r: f32 = image.get(&[0, y, x])?.clone().into();
            let g: f32 = image.get(&[1, y, x])?.clone().into();
            let b: f32 = image.get(&[2, y, x])?.clone().into();

            let gray = r * constants::RGB_TO_GRAY_WEIGHTS[0]
                + g * constants::RGB_TO_GRAY_WEIGHTS[1]
                + b * constants::RGB_TO_GRAY_WEIGHTS[2];

            result.set(&[0, y, x], gray.into())?;
        }
    }

    Ok(result)
}

/// Convert RGB image to HSV color space
pub fn rgb_to_hsv(image: &Tensor<f32>) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if channels != 3 {
        return Err(VisionError::InvalidArgument(
            "RGB to HSV conversion requires 3-channel input".to_string(),
        ));
    }

    let result = zeros_mut(&[3, height, width]);

    for y in 0..height {
        for x in 0..width {
            let r: f32 = image.get(&[0, y, x])?.clone().into();
            let g: f32 = image.get(&[1, y, x])?.clone().into();
            let b: f32 = image.get(&[2, y, x])?.clone().into();

            let (h, s, v) = rgb_to_hsv_pixel(r, g, b);

            result.set(&[0, y, x], h.into())?;
            result.set(&[1, y, x], s.into())?;
            result.set(&[2, y, x], v.into())?;
        }
    }

    Ok(result)
}

/// Convert HSV image to RGB color space
pub fn hsv_to_rgb(image: &Tensor<f32>) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if channels != 3 {
        return Err(VisionError::InvalidArgument(
            "HSV to RGB conversion requires 3-channel input".to_string(),
        ));
    }

    let result = zeros_mut(&[3, height, width]);

    for y in 0..height {
        for x in 0..width {
            let h: f32 = image.get(&[0, y, x])?.clone().into();
            let s: f32 = image.get(&[1, y, x])?.clone().into();
            let v: f32 = image.get(&[2, y, x])?.clone().into();

            let (r, g, b) = hsv_to_rgb_pixel(h, s, v);

            result.set(&[0, y, x], r.into())?;
            result.set(&[1, y, x], g.into())?;
            result.set(&[2, y, x], b.into())?;
        }
    }

    Ok(result)
}

/// Convert RGB to YUV color space
pub fn rgb_to_yuv(image: &Tensor<f32>) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if channels != 3 {
        return Err(VisionError::InvalidArgument(
            "RGB to YUV conversion requires 3-channel input".to_string(),
        ));
    }

    let result = zeros_mut(&[3, height, width]);

    for y in 0..height {
        for x in 0..width {
            let r: f32 = image.get(&[0, y, x])?.clone().into();
            let g: f32 = image.get(&[1, y, x])?.clone().into();
            let b: f32 = image.get(&[2, y, x])?.clone().into();

            let (y_val, u_val, v_val) = rgb_to_yuv_pixel(r, g, b);

            result.set(&[0, y, x], y_val.into())?;
            result.set(&[1, y, x], u_val.into())?;
            result.set(&[2, y, x], v_val.into())?;
        }
    }

    Ok(result)
}

/// Normalize image with specified configuration
pub fn normalize(image: &Tensor<f32>, config: NormalizationConfig) -> Result<Tensor<f32>> {
    match config.method {
        NormalizationMethod::MinMax => normalize_min_max(image, config.per_channel),
        NormalizationMethod::ZScore => normalize_zscore(image, config.per_channel, config.eps),
        NormalizationMethod::ImageNet => normalize_imagenet(image),
        NormalizationMethod::Custom => {
            let mean = config.mean.ok_or_else(|| {
                VisionError::InvalidArgument(
                    "Custom normalization requires mean values".to_string(),
                )
            })?;
            let std = config.std.ok_or_else(|| {
                VisionError::InvalidArgument("Custom normalization requires std values".to_string())
            })?;
            normalize_custom(image, &mean, &std, config.eps)
        }
    }
}

/// Normalize image using ImageNet statistics
pub fn normalize_imagenet(image: &Tensor<f32>) -> Result<Tensor<f32>> {
    let mean = constants::IMAGENET_MEAN.to_vec();
    let std = constants::IMAGENET_STD.to_vec();
    normalize_custom(image, &mean, &std, 1e-8)
}

/// Apply histogram equalization to improve contrast
pub fn histogram_equalization(image: &Tensor<f32>) -> Result<Tensor<f32>> {
    let config = HistogramConfig::default();
    histogram_equalization_with_config(image, config)
}

/// Apply histogram equalization with custom configuration
pub fn histogram_equalization_with_config(
    image: &Tensor<f32>,
    config: HistogramConfig,
) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    match config.method {
        HistogramMethod::Global => {
            apply_global_histogram_equalization(image, channels, height, width, config.bins)
        }
        HistogramMethod::Adaptive => {
            apply_adaptive_histogram_equalization(image, channels, height, width, &config)
        }
        HistogramMethod::Local => {
            apply_local_histogram_equalization(image, channels, height, width, &config)
        }
    }
}

/// Adjust image brightness
pub fn adjust_brightness(image: &Tensor<f32>, factor: f32) -> Result<Tensor<f32>> {
    let shape = image.shape();
    let dims = shape.dims();
    let result = zeros_mut(&dims);

    let total_elements = dims.iter().product::<usize>();

    for i in 0..total_elements {
        let indices = linear_to_indices(i, &dims);
        let pixel_val: f32 = image.get(&indices)?.clone().into();
        let adjusted = (pixel_val + factor).max(0.0).min(1.0);
        result.set(&indices, adjusted.into())?;
    }

    Ok(result)
}

/// Adjust image contrast
pub fn adjust_contrast(image: &Tensor<f32>, factor: f32) -> Result<Tensor<f32>> {
    let shape = image.shape();
    let dims = shape.dims();
    let result = zeros_mut(&dims);

    let total_elements = dims.iter().product::<usize>();

    for i in 0..total_elements {
        let indices = linear_to_indices(i, &dims);
        let pixel_val: f32 = image.get(&indices)?.clone().into();
        let adjusted = ((pixel_val - 0.5) * factor + 0.5).max(0.0).min(1.0);
        result.set(&indices, adjusted.into())?;
    }

    Ok(result)
}

/// Adjust image saturation (works on HSV)
pub fn adjust_saturation(image: &Tensor<f32>, factor: f32) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if channels != 3 {
        return Err(VisionError::InvalidArgument(
            "Saturation adjustment requires 3-channel RGB input".to_string(),
        ));
    }

    // Convert to HSV, adjust saturation, convert back to RGB
    let hsv = rgb_to_hsv(image)?;
    let adjusted_hsv = hsv.clone();

    for y in 0..height {
        for x in 0..width {
            let s: f32 = hsv.get(&[1, y, x])?.clone().into();
            let adjusted_s = (s * factor).max(0.0).min(1.0);
            adjusted_hsv.set(&[1, y, x], adjusted_s.into())?;
        }
    }

    hsv_to_rgb(&adjusted_hsv)
}

/// Adjust image hue (works on HSV)
pub fn adjust_hue(image: &Tensor<f32>, delta: f32) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if channels != 3 {
        return Err(VisionError::InvalidArgument(
            "Hue adjustment requires 3-channel RGB input".to_string(),
        ));
    }

    // Convert to HSV, adjust hue, convert back to RGB
    let hsv = rgb_to_hsv(image)?;
    let adjusted_hsv = hsv.clone();

    for y in 0..height {
        for x in 0..width {
            let h: f32 = hsv.get(&[0, y, x])?.clone().into();
            let adjusted_h = (h + delta) % 360.0;
            let adjusted_h = if adjusted_h < 0.0 {
                adjusted_h + 360.0
            } else {
                adjusted_h
            };
            adjusted_hsv.set(&[0, y, x], adjusted_h.into())?;
        }
    }

    hsv_to_rgb(&adjusted_hsv)
}

/// Apply gamma correction
pub fn gamma_correction(image: &Tensor<f32>, gamma: f32) -> Result<Tensor<f32>> {
    let shape = image.shape();
    let dims = shape.dims();
    let result = zeros_mut(&dims);

    let total_elements = dims.iter().product::<usize>();

    for i in 0..total_elements {
        let indices = linear_to_indices(i, &dims);
        let pixel_val: f32 = image.get(&indices)?.clone().into();
        let corrected = pixel_val.powf(1.0 / gamma);
        result.set(&indices, corrected.into())?;
    }

    Ok(result)
}

/// Extract a specific color channel
pub fn extract_channel(image: &Tensor<f32>, channel: usize) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if channel >= channels {
        return Err(VisionError::InvalidArgument(format!(
            "Channel {} does not exist in {}-channel image",
            channel, channels
        )));
    }

    let result = zeros_mut(&[1, height, width]);

    for y in 0..height {
        for x in 0..width {
            let pixel_val: f32 = image.get(&[channel, y, x])?.clone().into();
            result.set(&[0, y, x], pixel_val.into())?;
        }
    }

    Ok(result)
}

/// Compute color histogram
pub fn compute_histogram(image: &Tensor<f32>, bins: usize) -> Result<Vec<Vec<usize>>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;
    let mut histograms = vec![vec![0; bins]; channels];

    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let pixel_val: f32 = image.get(&[c, y, x])?.clone().into();
                let bin_idx = ((pixel_val.max(0.0).min(1.0) * (bins - 1) as f32).round() as usize)
                    .min(bins - 1);
                histograms[c][bin_idx] += 1;
            }
        }
    }

    Ok(histograms)
}

// Internal implementation functions

fn rgb_to_hsv_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max == 0.0 { 0.0 } else { delta / max };

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    (h, s, v)
}

fn hsv_to_rgb_pixel(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r_prime, g_prime, b_prime) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (r_prime + m, g_prime + m, b_prime + m)
}

fn rgb_to_yuv_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let u = -0.14713 * r - 0.28886 * g + 0.436 * b;
    let v = 0.615 * r - 0.51499 * g - 0.10001 * b;

    (y, u, v)
}

fn normalize_min_max(image: &Tensor<f32>, per_channel: bool) -> Result<Tensor<f32>> {
    let shape = image.shape();
    let dims = shape.dims();
    let result = zeros_mut(&dims);

    if per_channel && dims.len() == 3 {
        let (channels, height, width) = (dims[0], dims[1], dims[2]);

        for c in 0..channels {
            // Find min and max for this channel
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;

            for y in 0..height {
                for x in 0..width {
                    let val: f32 = image.get(&[c, y, x])?.clone().into();
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }

            let range = max_val - min_val;
            if range > 0.0 {
                for y in 0..height {
                    for x in 0..width {
                        let val: f32 = image.get(&[c, y, x])?.clone().into();
                        let normalized = (val - min_val) / range;
                        result.set(&[c, y, x], normalized.into())?;
                    }
                }
            }
        }
    } else {
        // Global normalization
        let total_elements = dims.iter().product::<usize>();
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for i in 0..total_elements {
            let indices = linear_to_indices(i, &dims);
            let val: f32 = image.get(&indices)?.clone().into();
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        let range = max_val - min_val;
        if range > 0.0 {
            for i in 0..total_elements {
                let indices = linear_to_indices(i, &dims);
                let val: f32 = image.get(&indices)?.clone().into();
                let normalized = (val - min_val) / range;
                result.set(&indices, normalized.into())?;
            }
        }
    }

    Ok(result)
}

fn normalize_zscore(image: &Tensor<f32>, per_channel: bool, eps: f32) -> Result<Tensor<f32>> {
    let shape = image.shape();
    let dims = shape.dims();
    let result = zeros_mut(&dims);

    if per_channel && dims.len() == 3 {
        let (channels, height, width) = (dims[0], dims[1], dims[2]);

        for c in 0..channels {
            // Compute mean
            let mut sum = 0.0;
            let num_pixels = height * width;

            for y in 0..height {
                for x in 0..width {
                    let val: f32 = image.get(&[c, y, x])?.clone().into();
                    sum += val;
                }
            }

            let mean = sum / num_pixels as f32;

            // Compute variance
            let mut sum_sq_diff = 0.0;
            for y in 0..height {
                for x in 0..width {
                    let val: f32 = image.get(&[c, y, x])?.clone().into();
                    sum_sq_diff += (val - mean).powi(2);
                }
            }

            let variance = sum_sq_diff / num_pixels as f32;
            let std = (variance + eps).sqrt();

            // Normalize
            for y in 0..height {
                for x in 0..width {
                    let val: f32 = image.get(&[c, y, x])?.clone().into();
                    let normalized = (val - mean) / std;
                    result.set(&[c, y, x], normalized.into())?;
                }
            }
        }
    } else {
        // Global normalization
        let total_elements = dims.iter().product::<usize>();

        // Compute mean
        let mut sum = 0.0;
        for i in 0..total_elements {
            let indices = linear_to_indices(i, &dims);
            let val: f32 = image.get(&indices)?.clone().into();
            sum += val;
        }
        let mean = sum / total_elements as f32;

        // Compute variance
        let mut sum_sq_diff = 0.0;
        for i in 0..total_elements {
            let indices = linear_to_indices(i, &dims);
            let val: f32 = image.get(&indices)?.clone().into();
            sum_sq_diff += (val - mean).powi(2);
        }
        let variance = sum_sq_diff / total_elements as f32;
        let std = (variance + eps).sqrt();

        // Normalize
        for i in 0..total_elements {
            let indices = linear_to_indices(i, &dims);
            let val: f32 = image.get(&indices)?.clone().into();
            let normalized = (val - mean) / std;
            result.set(&indices, normalized.into())?;
        }
    }

    Ok(result)
}

fn normalize_custom(
    image: &Tensor<f32>,
    mean: &[f32],
    std: &[f32],
    eps: f32,
) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if mean.len() != channels || std.len() != channels {
        return Err(VisionError::InvalidArgument(
            "Mean and std must have same length as number of channels".to_string(),
        ));
    }

    let result = zeros_mut(&[channels, height, width]);

    for c in 0..channels {
        let channel_mean = mean[c];
        let channel_std = std[c].max(eps);

        for y in 0..height {
            for x in 0..width {
                let val: f32 = image.get(&[c, y, x])?.clone().into();
                let normalized = (val - channel_mean) / channel_std;
                result.set(&[c, y, x], normalized.into())?;
            }
        }
    }

    Ok(result)
}

fn apply_global_histogram_equalization(
    image: &Tensor<f32>,
    channels: usize,
    height: usize,
    width: usize,
    bins: usize,
) -> Result<Tensor<f32>> {
    let result = zeros_mut(&[channels, height, width]);

    for c in 0..channels {
        // Compute histogram
        let mut histogram = vec![0; bins];
        for y in 0..height {
            for x in 0..width {
                let val: f32 = image.get(&[c, y, x])?.clone().into();
                let bin_idx =
                    ((val.max(0.0).min(1.0) * (bins - 1) as f32).round() as usize).min(bins - 1);
                histogram[bin_idx] += 1;
            }
        }

        // Compute cumulative distribution function
        let mut cdf = vec![0.0; bins];
        cdf[0] = histogram[0] as f32;
        for i in 1..bins {
            cdf[i] = cdf[i - 1] + histogram[i] as f32;
        }

        // Normalize CDF
        let total_pixels = (height * width) as f32;
        for i in 0..bins {
            cdf[i] /= total_pixels;
        }

        // Apply equalization
        for y in 0..height {
            for x in 0..width {
                let val: f32 = image.get(&[c, y, x])?.clone().into();
                let bin_idx =
                    ((val.max(0.0).min(1.0) * (bins - 1) as f32).round() as usize).min(bins - 1);
                let equalized = cdf[bin_idx];
                result.set(&[c, y, x], equalized.into())?;
            }
        }
    }

    Ok(result)
}

fn apply_adaptive_histogram_equalization(
    image: &Tensor<f32>,
    channels: usize,
    height: usize,
    width: usize,
    config: &HistogramConfig,
) -> Result<Tensor<f32>> {
    let result = image.clone();

    let (tile_height, tile_width) = config.tile_size;
    let y_tiles = (height + tile_height - 1) / tile_height;
    let x_tiles = (width + tile_width - 1) / tile_width;

    for c in 0..channels {
        for ty in 0..y_tiles {
            for tx in 0..x_tiles {
                let y_start = ty * tile_height;
                let y_end = (y_start + tile_height).min(height);
                let x_start = tx * tile_width;
                let x_end = (x_start + tile_width).min(width);

                // Extract tile
                let mut tile_values = Vec::new();
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let val: f32 = image.get(&[c, y, x])?.clone().into();
                        tile_values.push(val);
                    }
                }

                // Apply CLAHE to tile
                let equalized_tile =
                    apply_clahe_to_tile(&tile_values, config.bins, config.clip_limit)?;

                // Write back to result
                let mut idx = 0;
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        result.set(&[c, y, x], equalized_tile[idx].into())?;
                        idx += 1;
                    }
                }
            }
        }
    }

    Ok(result)
}

fn apply_local_histogram_equalization(
    image: &Tensor<f32>,
    channels: usize,
    height: usize,
    width: usize,
    config: &HistogramConfig,
) -> Result<Tensor<f32>> {
    // Simplified local histogram equalization using sliding window
    let window_size = config.tile_size.0.min(config.tile_size.1);
    let result = zeros_mut(&[channels, height, width]);

    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let y_start = (y as i32 - window_size as i32 / 2).max(0) as usize;
                let y_end = (y + window_size / 2 + 1).min(height);
                let x_start = (x as i32 - window_size as i32 / 2).max(0) as usize;
                let x_end = (x + window_size / 2 + 1).min(width);

                // Collect local window values
                let mut local_values = Vec::new();
                for ly in y_start..y_end {
                    for lx in x_start..x_end {
                        let val: f32 = image.get(&[c, ly, lx])?.clone().into();
                        local_values.push(val);
                    }
                }

                // Apply local equalization
                local_values.sort_by(|a, b| a.partial_cmp(b).expect("comparison should succeed"));
                let current_val: f32 = image.get(&[c, y, x])?.clone().into();

                let rank = local_values
                    .iter()
                    .position(|&v| v >= current_val)
                    .unwrap_or(local_values.len() - 1);
                let equalized = rank as f32 / (local_values.len() - 1) as f32;

                result.set(&[c, y, x], equalized.into())?;
            }
        }
    }

    Ok(result)
}

fn apply_clahe_to_tile(values: &[f32], bins: usize, clip_limit: f32) -> Result<Vec<f32>> {
    // Compute histogram
    let mut histogram = vec![0; bins];
    for &val in values {
        let bin_idx = ((val.max(0.0).min(1.0) * (bins - 1) as f32).round() as usize).min(bins - 1);
        histogram[bin_idx] += 1;
    }

    // Apply clipping
    let max_count = (clip_limit * values.len() as f32) as usize;
    let mut excess = 0;
    for count in &mut histogram {
        if *count > max_count {
            excess += *count - max_count;
            *count = max_count;
        }
    }

    // Redistribute excess uniformly
    let redistribution = excess / bins;
    for count in &mut histogram {
        *count += redistribution;
    }

    // Compute CDF
    let mut cdf = vec![0.0; bins];
    cdf[0] = histogram[0] as f32;
    for i in 1..bins {
        cdf[i] = cdf[i - 1] + histogram[i] as f32;
    }

    // Normalize CDF
    let total = values.len() as f32;
    for i in 0..bins {
        cdf[i] /= total;
    }

    // Apply equalization
    let mut result = Vec::with_capacity(values.len());
    for &val in values {
        let bin_idx = ((val.max(0.0).min(1.0) * (bins - 1) as f32).round() as usize).min(bins - 1);
        result.push(cdf[bin_idx]);
    }

    Ok(result)
}

fn linear_to_indices(linear_index: usize, dims: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; dims.len()];
    let mut remaining = linear_index;

    for i in (0..dims.len()).rev() {
        indices[i] = remaining % dims[i];
        remaining /= dims[i];
    }

    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_rgb_to_grayscale() -> Result<()> {
        let image = zeros(&[3, 32, 32])?;
        let result = rgb_to_grayscale(&image)?;

        assert_eq!(result.shape().dims(), &[1, 32, 32]);
        Ok(())
    }

    #[test]
    fn test_rgb_hsv_conversion() -> Result<()> {
        let rgb_image = zeros(&[3, 16, 16])?;
        let hsv_image = rgb_to_hsv(&rgb_image)?;
        let back_to_rgb = hsv_to_rgb(&hsv_image)?;

        assert_eq!(hsv_image.shape().dims(), &[3, 16, 16]);
        assert_eq!(back_to_rgb.shape().dims(), &[3, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_normalization_methods() -> Result<()> {
        let image = zeros(&[3, 16, 16])?;

        // Test min-max normalization
        let config = NormalizationConfig {
            method: NormalizationMethod::MinMax,
            ..Default::default()
        };
        let result = normalize(&image, config)?;
        assert_eq!(result.shape().dims(), &[3, 16, 16]);

        // Test z-score normalization
        let config = NormalizationConfig::zscore();
        let result = normalize(&image, config)?;
        assert_eq!(result.shape().dims(), &[3, 16, 16]);

        // Test ImageNet normalization
        let result = normalize_imagenet(&image)?;
        assert_eq!(result.shape().dims(), &[3, 16, 16]);

        Ok(())
    }

    #[test]
    fn test_histogram_equalization() -> Result<()> {
        let image = zeros(&[1, 32, 32])?;
        let result = histogram_equalization(&image)?;

        assert_eq!(result.shape().dims(), &[1, 32, 32]);
        Ok(())
    }

    #[test]
    fn test_color_adjustments() -> Result<()> {
        let image = zeros(&[3, 16, 16])?;

        let brightness_adjusted = adjust_brightness(&image, 0.1)?;
        assert_eq!(brightness_adjusted.shape().dims(), &[3, 16, 16]);

        let contrast_adjusted = adjust_contrast(&image, 1.2)?;
        assert_eq!(contrast_adjusted.shape().dims(), &[3, 16, 16]);

        let saturation_adjusted = adjust_saturation(&image, 0.8)?;
        assert_eq!(saturation_adjusted.shape().dims(), &[3, 16, 16]);

        let hue_adjusted = adjust_hue(&image, 30.0)?;
        assert_eq!(hue_adjusted.shape().dims(), &[3, 16, 16]);

        Ok(())
    }

    #[test]
    fn test_gamma_correction() -> Result<()> {
        let image = zeros(&[3, 16, 16])?;
        let result = gamma_correction(&image, 2.2)?;

        assert_eq!(result.shape().dims(), &[3, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_channel_extraction() -> Result<()> {
        let image = zeros(&[3, 16, 16])?;
        let red_channel = extract_channel(&image, 0)?;

        assert_eq!(red_channel.shape().dims(), &[1, 16, 16]);

        // Test invalid channel
        assert!(extract_channel(&image, 5).is_err());
        Ok(())
    }

    #[test]
    fn test_compute_histogram() -> Result<()> {
        let image = zeros(&[3, 16, 16])?;
        let histograms = compute_histogram(&image, 256)?;

        assert_eq!(histograms.len(), 3); // 3 channels
        assert_eq!(histograms[0].len(), 256); // 256 bins
        Ok(())
    }

    #[test]
    fn test_color_space_configs() -> Result<()> {
        let rgb_config = ColorConfig::default();
        assert_eq!(rgb_config.target_space, ColorSpace::RGB);

        let gray_config = ColorConfig::rgb_to_grayscale();
        assert_eq!(gray_config.target_space, ColorSpace::Grayscale);

        let hsv_config = ColorConfig::hsv();
        assert_eq!(hsv_config.target_space, ColorSpace::HSV);

        Ok(())
    }

    #[test]
    fn test_normalization_configs() -> Result<()> {
        let imagenet_config = NormalizationConfig::imagenet();
        assert_eq!(imagenet_config.method, NormalizationMethod::ImageNet);
        assert!(imagenet_config.mean.is_some());
        assert!(imagenet_config.std.is_some());

        let custom_config = NormalizationConfig::custom(vec![0.5, 0.5, 0.5], vec![0.2, 0.2, 0.2]);
        assert_eq!(custom_config.method, NormalizationMethod::Custom);

        Ok(())
    }

    #[test]
    fn test_pixel_conversions() -> Result<()> {
        // Test RGB to HSV pixel conversion
        let (h, s, v) = rgb_to_hsv_pixel(1.0, 0.0, 0.0); // Pure red
        assert!((h - 0.0).abs() < 1e-6);
        assert!((s - 1.0).abs() < 1e-6);
        assert!((v - 1.0).abs() < 1e-6);

        // Test HSV to RGB pixel conversion
        let (r, g, b) = hsv_to_rgb_pixel(0.0, 1.0, 1.0); // Pure red
        assert!((r - 1.0).abs() < 1e-6);
        assert!(g.abs() < 1e-6);
        assert!(b.abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_invalid_inputs() -> Result<()> {
        // Test RGB to grayscale with wrong channels
        let image = zeros(&[1, 32, 32])?;
        assert!(rgb_to_grayscale(&image).is_err());

        // Test saturation adjustment on single channel
        assert!(adjust_saturation(&image, 1.0).is_err());

        // Test custom normalization with mismatched dimensions
        let rgb_image = zeros(&[3, 16, 16])?;
        let config = NormalizationConfig::custom(vec![0.5], vec![0.2]); // Wrong size
        assert!(normalize(&rgb_image, config).is_err());

        Ok(())
    }
}
