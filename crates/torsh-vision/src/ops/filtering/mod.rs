//! Filtering and enhancement operations for computer vision
//!
//! This module provides comprehensive image filtering and enhancement operations including:
//! - Edge detection algorithms (Sobel, Canny, Laplacian, Prewitt)
//! - Gaussian blur and other smoothing filters
//! - Morphological operations (erosion, dilation, opening, closing)
//! - Kernel-based convolution operations
//! - Noise reduction and enhancement filters

use crate::ops::common::{
    constants, utils, EdgeDetectionAlgorithm, InterpolationMode, MorphologicalOperation,
    PaddingMode,
};
use crate::{Result, VisionError};
use torsh_tensor::creation::{full, ones, zeros};
use torsh_tensor::Tensor;

/// Configuration for filtering operations
#[derive(Debug, Clone)]
pub struct FilteringConfig {
    /// Padding mode for convolution operations
    pub padding: PaddingMode,
    /// Whether to normalize kernel weights
    pub normalize_kernel: bool,
    /// Clamping range for output values
    pub clamp_range: Option<(f32, f32)>,
}

impl Default for FilteringConfig {
    fn default() -> Self {
        Self {
            padding: PaddingMode::Zero,
            normalize_kernel: true,
            clamp_range: None,
        }
    }
}

impl FilteringConfig {
    /// Create configuration optimized for edge detection
    pub fn edge_detection() -> Self {
        Self {
            padding: PaddingMode::Reflect,
            normalize_kernel: false,
            clamp_range: Some((0.0, 255.0)),
        }
    }

    /// Create configuration optimized for smoothing operations
    pub fn smoothing() -> Self {
        Self {
            padding: PaddingMode::Reflect,
            normalize_kernel: true,
            clamp_range: None,
        }
    }

    /// Set padding mode
    pub fn with_padding(mut self, padding: PaddingMode) -> Self {
        self.padding = padding;
        self
    }

    /// Set kernel normalization
    pub fn with_normalize_kernel(mut self, normalize: bool) -> Self {
        self.normalize_kernel = normalize;
        self
    }

    /// Set output clamping range
    pub fn with_clamp_range(mut self, range: Option<(f32, f32)>) -> Self {
        self.clamp_range = range;
        self
    }
}

/// Edge detection configuration
#[derive(Debug, Clone)]
pub struct EdgeDetectionConfig {
    /// Algorithm to use for edge detection
    pub algorithm: EdgeDetectionAlgorithm,
    /// Low threshold for Canny edge detection
    pub low_threshold: f32,
    /// High threshold for Canny edge detection
    pub high_threshold: f32,
    /// Gaussian blur sigma for preprocessing
    pub blur_sigma: f32,
    /// Kernel size for gradient calculation
    pub kernel_size: usize,
}

impl Default for EdgeDetectionConfig {
    fn default() -> Self {
        Self {
            algorithm: EdgeDetectionAlgorithm::Sobel,
            low_threshold: constants::DEFAULT_CANNY_LOW_THRESHOLD,
            high_threshold: constants::DEFAULT_CANNY_HIGH_THRESHOLD,
            blur_sigma: constants::DEFAULT_GAUSSIAN_SIGMA,
            kernel_size: 3,
        }
    }
}

impl EdgeDetectionConfig {
    /// Create Sobel edge detection configuration
    pub fn sobel() -> Self {
        Self {
            algorithm: EdgeDetectionAlgorithm::Sobel,
            ..Default::default()
        }
    }

    /// Create Canny edge detection configuration
    pub fn canny(low_threshold: f32, high_threshold: f32) -> Self {
        Self {
            algorithm: EdgeDetectionAlgorithm::Canny,
            low_threshold,
            high_threshold,
            ..Default::default()
        }
    }

    /// Create Laplacian edge detection configuration
    pub fn laplacian() -> Self {
        Self {
            algorithm: EdgeDetectionAlgorithm::Laplacian,
            ..Default::default()
        }
    }

    /// Create Prewitt edge detection configuration
    pub fn prewitt() -> Self {
        Self {
            algorithm: EdgeDetectionAlgorithm::Prewitt,
            ..Default::default()
        }
    }
}

/// Gaussian blur configuration
#[derive(Debug, Clone)]
pub struct GaussianBlurConfig {
    /// Sigma value for Gaussian kernel
    pub sigma: f32,
    /// Kernel size (must be odd)
    pub kernel_size: usize,
    /// Padding mode for convolution
    pub padding: PaddingMode,
}

impl Default for GaussianBlurConfig {
    fn default() -> Self {
        Self {
            sigma: constants::DEFAULT_GAUSSIAN_SIGMA,
            kernel_size: 5,
            padding: PaddingMode::Reflect,
        }
    }
}

impl GaussianBlurConfig {
    /// Create configuration with custom sigma
    pub fn with_sigma(sigma: f32) -> Self {
        let kernel_size = ((sigma * 6.0).ceil() as usize).max(3) | 1; // Ensure odd
        Self {
            sigma,
            kernel_size,
            ..Default::default()
        }
    }

    /// Create configuration with custom kernel size
    pub fn with_kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = if kernel_size % 2 == 0 {
            kernel_size + 1
        } else {
            kernel_size
        };
        self
    }
}

/// Morphological operations configuration
#[derive(Debug, Clone)]
pub struct MorphologyConfig {
    /// Type of morphological operation
    pub operation: MorphologicalOperation,
    /// Structuring element size
    pub kernel_size: usize,
    /// Number of iterations to apply the operation
    pub iterations: usize,
}

impl Default for MorphologyConfig {
    fn default() -> Self {
        Self {
            operation: MorphologicalOperation::Erosion,
            kernel_size: 3,
            iterations: 1,
        }
    }
}

impl MorphologyConfig {
    /// Create erosion configuration
    pub fn erosion(kernel_size: usize) -> Self {
        Self {
            operation: MorphologicalOperation::Erosion,
            kernel_size,
            iterations: 1,
        }
    }

    /// Create dilation configuration
    pub fn dilation(kernel_size: usize) -> Self {
        Self {
            operation: MorphologicalOperation::Dilation,
            kernel_size,
            iterations: 1,
        }
    }

    /// Create opening configuration (erosion followed by dilation)
    pub fn opening(kernel_size: usize) -> Self {
        Self {
            operation: MorphologicalOperation::Opening,
            kernel_size,
            iterations: 1,
        }
    }

    /// Create closing configuration (dilation followed by erosion)
    pub fn closing(kernel_size: usize) -> Self {
        Self {
            operation: MorphologicalOperation::Closing,
            kernel_size,
            iterations: 1,
        }
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }
}

/// Apply Sobel edge detection to an image
pub fn sobel_edge_detection(image: &Tensor<f32>) -> Result<Tensor<f32>> {
    let config = EdgeDetectionConfig::sobel();
    edge_detection(image, config)
}

/// Apply edge detection with custom configuration
pub fn edge_detection(image: &Tensor<f32>, config: EdgeDetectionConfig) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    match config.algorithm {
        EdgeDetectionAlgorithm::Sobel => apply_sobel_filter(image, channels, height, width),
        EdgeDetectionAlgorithm::Canny => {
            apply_canny_filter(image, channels, height, width, &config)
        }
        EdgeDetectionAlgorithm::Laplacian => apply_laplacian_filter(image, channels, height, width),
        EdgeDetectionAlgorithm::Prewitt => apply_prewitt_filter(image, channels, height, width),
    }
}

/// Apply Gaussian blur to an image
pub fn gaussian_blur(image: &Tensor<f32>, sigma: f32) -> Result<Tensor<f32>> {
    let config = GaussianBlurConfig::with_sigma(sigma);
    gaussian_blur_with_config(image, config)
}

/// Apply Gaussian blur with custom configuration
pub fn gaussian_blur_with_config(
    image: &Tensor<f32>,
    config: GaussianBlurConfig,
) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    // Generate Gaussian kernel
    let kernel_1d = utils::gaussian_kernel_1d(config.sigma, config.kernel_size);

    // Apply separable convolution (horizontal then vertical)
    let temp_result =
        apply_horizontal_convolution(image, &kernel_1d, channels, height, width, config.padding)?;
    apply_vertical_convolution(
        &temp_result,
        &kernel_1d,
        channels,
        height,
        width,
        config.padding,
    )
}

/// Apply morphological operations to a binary or grayscale image
pub fn morphological_operation(
    image: &Tensor<f32>,
    config: MorphologyConfig,
) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if channels != 1 {
        return Err(VisionError::InvalidArgument(
            "Morphological operations require single-channel (grayscale) images".to_string(),
        ));
    }

    let structuring_element = utils::create_structuring_element(config.kernel_size);
    let mut result = image.clone();

    for _ in 0..config.iterations {
        result = match config.operation {
            MorphologicalOperation::Erosion => {
                apply_erosion(&result, &structuring_element, height, width)?
            }
            MorphologicalOperation::Dilation => {
                apply_dilation(&result, &structuring_element, height, width)?
            }
            MorphologicalOperation::Opening => {
                let eroded = apply_erosion(&result, &structuring_element, height, width)?;
                apply_dilation(&eroded, &structuring_element, height, width)?
            }
            MorphologicalOperation::Closing => {
                let dilated = apply_dilation(&result, &structuring_element, height, width)?;
                apply_erosion(&dilated, &structuring_element, height, width)?
            }
            MorphologicalOperation::Gradient => {
                let dilated = apply_dilation(&result, &structuring_element, height, width)?;
                let eroded = apply_erosion(&result, &structuring_element, height, width)?;
                subtract_tensors(&dilated, &eroded)?
            }
            MorphologicalOperation::TopHat => {
                let opened = apply_erosion(&result, &structuring_element, height, width)?;
                let opened = apply_dilation(&opened, &structuring_element, height, width)?;
                subtract_tensors(&result, &opened)?
            }
            MorphologicalOperation::BlackHat => {
                let closed = apply_dilation(&result, &structuring_element, height, width)?;
                let closed = apply_erosion(&closed, &structuring_element, height, width)?;
                subtract_tensors(&closed, &result)?
            }
        };
    }

    Ok(result)
}

/// Apply a custom convolution kernel to an image
pub fn conv2d(image: &Tensor<f32>, kernel: &Tensor<f32>) -> Result<Tensor<f32>> {
    conv2d_with_config(image, kernel, FilteringConfig::default())
}

/// Apply convolution with custom configuration
pub fn conv2d_with_config(
    image: &Tensor<f32>,
    kernel: &Tensor<f32>,
    config: FilteringConfig,
) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;
    let kernel_shape = kernel.shape();
    let kernel_dims = kernel_shape.dims();

    if kernel_dims.len() != 2 {
        return Err(VisionError::InvalidShape(
            "Convolution kernel must be 2D".to_string(),
        ));
    }

    let (kernel_height, kernel_width) = (kernel_dims[0], kernel_dims[1]);

    if kernel_height % 2 == 0 || kernel_width % 2 == 0 {
        return Err(VisionError::InvalidArgument(
            "Kernel dimensions must be odd".to_string(),
        ));
    }

    apply_2d_convolution(
        image,
        kernel,
        channels,
        height,
        width,
        kernel_height,
        kernel_width,
        config,
    )
}

/// Apply median filter for noise reduction
pub fn median_filter(image: &Tensor<f32>, kernel_size: usize) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if kernel_size % 2 == 0 {
        return Err(VisionError::InvalidArgument(
            "Kernel size must be odd for median filter".to_string(),
        ));
    }

    apply_median_filter(image, channels, height, width, kernel_size)
}

/// Apply bilateral filter for edge-preserving smoothing
pub fn bilateral_filter(
    image: &Tensor<f32>,
    sigma_spatial: f32,
    sigma_color: f32,
    kernel_size: usize,
) -> Result<Tensor<f32>> {
    let (channels, height, width) = utils::validate_image_tensor_3d(image)?;

    if kernel_size % 2 == 0 {
        return Err(VisionError::InvalidArgument(
            "Kernel size must be odd for bilateral filter".to_string(),
        ));
    }

    apply_bilateral_filter(
        image,
        channels,
        height,
        width,
        kernel_size,
        sigma_spatial,
        sigma_color,
    )
}

// Internal implementation functions

fn apply_sobel_filter(
    image: &Tensor<f32>,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Tensor<f32>> {
    let sobel_x = tensor_from_2d_array(&constants::SOBEL_X_KERNEL)?;
    let sobel_y = tensor_from_2d_array(&constants::SOBEL_Y_KERNEL)?;

    let grad_x = apply_2d_convolution(
        image,
        &sobel_x,
        channels,
        height,
        width,
        3,
        3,
        FilteringConfig::edge_detection(),
    )?;

    let grad_y = apply_2d_convolution(
        image,
        &sobel_y,
        channels,
        height,
        width,
        3,
        3,
        FilteringConfig::edge_detection(),
    )?;

    // Compute gradient magnitude
    compute_gradient_magnitude(&grad_x, &grad_y)
}

fn apply_canny_filter(
    image: &Tensor<f32>,
    channels: usize,
    height: usize,
    width: usize,
    config: &EdgeDetectionConfig,
) -> Result<Tensor<f32>> {
    // Step 1: Apply Gaussian blur
    let blur_config = GaussianBlurConfig::with_sigma(config.blur_sigma);
    let blurred = gaussian_blur_with_config(image, blur_config)?;

    // Step 2: Compute gradients
    let grad_x = apply_sobel_filter(&blurred, channels, height, width)?;
    let sobel_y = tensor_from_2d_array(&constants::SOBEL_Y_KERNEL)?;
    let grad_y = apply_2d_convolution(
        &blurred,
        &sobel_y,
        channels,
        height,
        width,
        3,
        3,
        FilteringConfig::edge_detection(),
    )?;

    // Step 3: Non-maximum suppression
    let magnitude = compute_gradient_magnitude(&grad_x, &grad_y)?;
    let direction = compute_gradient_direction(&grad_x, &grad_y)?;
    let suppressed = non_maximum_suppression(&magnitude, &direction)?;

    // Step 4: Double thresholding and edge tracking
    double_threshold_and_hysteresis(&suppressed, config.low_threshold, config.high_threshold)
}

fn apply_laplacian_filter(
    image: &Tensor<f32>,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Tensor<f32>> {
    // Laplacian kernel for edge detection
    let laplacian_kernel = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]];

    let kernel = tensor_from_2d_array(&laplacian_kernel)?;
    apply_2d_convolution(
        image,
        &kernel,
        channels,
        height,
        width,
        3,
        3,
        FilteringConfig::edge_detection(),
    )
}

fn apply_prewitt_filter(
    image: &Tensor<f32>,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Tensor<f32>> {
    // Prewitt kernels
    let prewitt_x = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let prewitt_y = [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];

    let kernel_x = tensor_from_2d_array(&prewitt_x)?;
    let kernel_y = tensor_from_2d_array(&prewitt_y)?;

    let grad_x = apply_2d_convolution(
        image,
        &kernel_x,
        channels,
        height,
        width,
        3,
        3,
        FilteringConfig::edge_detection(),
    )?;
    let grad_y = apply_2d_convolution(
        image,
        &kernel_y,
        channels,
        height,
        width,
        3,
        3,
        FilteringConfig::edge_detection(),
    )?;

    compute_gradient_magnitude(&grad_x, &grad_y)
}

fn apply_horizontal_convolution(
    image: &Tensor<f32>,
    kernel: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    padding: PaddingMode,
) -> Result<Tensor<f32>> {
    let mut result = zeros(&[channels, height, width])?;
    let kernel_size = kernel.len();
    let kernel_radius = kernel_size / 2;

    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let mut sum: f32 = 0.0;

                for k in 0..kernel_size {
                    let src_x = x as i64 + k as i64 - kernel_radius as i64;
                    let clamped_x = apply_padding_1d(src_x, width, padding);

                    let pixel_val: f32 = image.get(&[c, y, clamped_x])?.clone().into();
                    sum += pixel_val * kernel[k];
                }

                result.set(&[c, y, x], sum.into())?;
            }
        }
    }

    Ok(result)
}

fn apply_vertical_convolution(
    image: &Tensor<f32>,
    kernel: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    padding: PaddingMode,
) -> Result<Tensor<f32>> {
    let mut result = zeros(&[channels, height, width])?;
    let kernel_size = kernel.len();
    let kernel_radius = kernel_size / 2;

    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let mut sum: f32 = 0.0;

                for k in 0..kernel_size {
                    let src_y = y as i64 + k as i64 - kernel_radius as i64;
                    let clamped_y = apply_padding_1d(src_y, height, padding);

                    let pixel_val: f32 = image.get(&[c, clamped_y, x])?.clone().into();
                    sum += pixel_val * kernel[k];
                }

                result.set(&[c, y, x], sum.into())?;
            }
        }
    }

    Ok(result)
}

fn apply_2d_convolution(
    image: &Tensor<f32>,
    kernel: &Tensor<f32>,
    channels: usize,
    height: usize,
    width: usize,
    kernel_height: usize,
    kernel_width: usize,
    config: FilteringConfig,
) -> Result<Tensor<f32>> {
    let mut result = zeros(&[channels, height, width])?;
    let kernel_radius_y = kernel_height / 2;
    let kernel_radius_x = kernel_width / 2;

    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let mut sum: f32 = 0.0;

                for ky in 0..kernel_height {
                    for kx in 0..kernel_width {
                        let src_y = y as i64 + ky as i64 - kernel_radius_y as i64;
                        let src_x = x as i64 + kx as i64 - kernel_radius_x as i64;

                        let clamped_y = apply_padding_1d(src_y, height, config.padding);
                        let clamped_x = apply_padding_1d(src_x, width, config.padding);

                        // Defensive bounds checking to prevent index out of bounds errors
                        if clamped_y >= height || clamped_x >= width {
                            continue; // Skip this kernel position if bounds would be exceeded
                        }

                        let pixel_val: f32 = image.get(&[c, clamped_y, clamped_x])?.clone().into();
                        let kernel_val: f32 = kernel.get(&[ky, kx])?.clone().into();
                        sum += pixel_val * kernel_val;
                    }
                }

                if let Some((min_val, max_val)) = config.clamp_range {
                    sum = sum.max(min_val).min(max_val);
                }

                result.set(&[c, y, x], sum.into())?;
            }
        }
    }

    Ok(result)
}

fn apply_erosion(
    image: &Tensor<f32>,
    structuring_element: &[Vec<bool>],
    height: usize,
    width: usize,
) -> Result<Tensor<f32>> {
    let mut result = zeros(&[1, height, width])?;
    let kernel_size = structuring_element.len();
    let kernel_radius = kernel_size / 2;

    for y in 0..height {
        for x in 0..width {
            let mut min_val = f32::INFINITY;

            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    if structuring_element[ky][kx] {
                        let src_y = y as i64 + ky as i64 - kernel_radius as i64;
                        let src_x = x as i64 + kx as i64 - kernel_radius as i64;

                        let clamped_y = utils::clamp_coord(src_y, 0, height);
                        let clamped_x = utils::clamp_coord(src_x, 0, width);

                        let pixel_val: f32 = image.get(&[0, clamped_y, clamped_x])?.clone().into();
                        min_val = min_val.min(pixel_val);
                    }
                }
            }

            result.set(&[0, y, x], min_val.into())?;
        }
    }

    Ok(result)
}

fn apply_dilation(
    image: &Tensor<f32>,
    structuring_element: &[Vec<bool>],
    height: usize,
    width: usize,
) -> Result<Tensor<f32>> {
    let mut result = zeros(&[1, height, width])?;
    let kernel_size = structuring_element.len();
    let kernel_radius = kernel_size / 2;

    for y in 0..height {
        for x in 0..width {
            let mut max_val = f32::NEG_INFINITY;

            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    if structuring_element[ky][kx] {
                        let src_y = y as i64 + ky as i64 - kernel_radius as i64;
                        let src_x = x as i64 + kx as i64 - kernel_radius as i64;

                        let clamped_y = utils::clamp_coord(src_y, 0, height);
                        let clamped_x = utils::clamp_coord(src_x, 0, width);

                        let pixel_val: f32 = image.get(&[0, clamped_y, clamped_x])?.clone().into();
                        max_val = max_val.max(pixel_val);
                    }
                }
            }

            result.set(&[0, y, x], max_val.into())?;
        }
    }

    Ok(result)
}

fn apply_median_filter(
    image: &Tensor<f32>,
    channels: usize,
    height: usize,
    width: usize,
    kernel_size: usize,
) -> Result<Tensor<f32>> {
    let mut result = zeros(&[channels, height, width])?;
    let kernel_radius = kernel_size / 2;

    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let mut values = Vec::new();

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let src_y = y as i64 + ky as i64 - kernel_radius as i64;
                        let src_x = x as i64 + kx as i64 - kernel_radius as i64;

                        let clamped_y = utils::clamp_coord(src_y, 0, height);
                        let clamped_x = utils::clamp_coord(src_x, 0, width);

                        let pixel_val: f32 = image.get(&[c, clamped_y, clamped_x])?.clone().into();
                        values.push(pixel_val);
                    }
                }

                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = values[values.len() / 2];
                result.set(&[c, y, x], median.into())?;
            }
        }
    }

    Ok(result)
}

fn apply_bilateral_filter(
    image: &Tensor<f32>,
    channels: usize,
    height: usize,
    width: usize,
    kernel_size: usize,
    sigma_spatial: f32,
    sigma_color: f32,
) -> Result<Tensor<f32>> {
    let mut result = zeros(&[channels, height, width])?;
    let kernel_radius = kernel_size / 2;
    let spatial_coeff = -1.0 / (2.0 * sigma_spatial * sigma_spatial);
    let color_coeff = -1.0 / (2.0 * sigma_color * sigma_color);

    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let center_val: f32 = image.get(&[c, y, x])?.clone().into();
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let src_y = y as i64 + ky as i64 - kernel_radius as i64;
                        let src_x = x as i64 + kx as i64 - kernel_radius as i64;

                        let clamped_y = utils::clamp_coord(src_y, 0, height);
                        let clamped_x = utils::clamp_coord(src_x, 0, width);

                        let pixel_val: f32 = image.get(&[c, clamped_y, clamped_x])?.clone().into();

                        // Spatial distance weight
                        let spatial_dist = ((ky as f32 - kernel_radius as f32).powi(2)
                            + (kx as f32 - kernel_radius as f32).powi(2));
                        let spatial_weight = (spatial_dist * spatial_coeff).exp();

                        // Color distance weight
                        let color_dist = (pixel_val - center_val).powi(2);
                        let color_weight = (color_dist * color_coeff).exp();

                        let total_weight = spatial_weight * color_weight;
                        weighted_sum += pixel_val * total_weight;
                        weight_sum += total_weight;
                    }
                }

                let filtered_val = if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    center_val
                };
                result.set(&[c, y, x], filtered_val.into())?;
            }
        }
    }

    Ok(result)
}

// Helper functions

fn apply_padding_1d(coord: i64, size: usize, padding: PaddingMode) -> usize {
    match padding {
        PaddingMode::Zero => utils::clamp_coord(coord, 0, size),
        PaddingMode::Reflect => {
            if coord < 0 {
                (-coord).min(size as i64 - 1) as usize
            } else if coord >= size as i64 {
                (2 * size as i64 - coord - 2).max(0) as usize
            } else {
                coord as usize
            }
        }
        PaddingMode::Replicate => utils::clamp_coord(coord, 0, size),
        PaddingMode::Circular => {
            let mut result = coord % size as i64;
            if result < 0 {
                result += size as i64;
            }
            result as usize
        }
    }
}

fn tensor_from_2d_array(array: &[[f32; 3]; 3]) -> Result<Tensor<f32>> {
    let mut result = zeros(&[3, 3])?;
    for i in 0..3 {
        for j in 0..3 {
            result.set(&[i, j], array[i][j].into())?;
        }
    }
    Ok(result)
}

fn compute_gradient_magnitude(grad_x: &Tensor<f32>, grad_y: &Tensor<f32>) -> Result<Tensor<f32>> {
    let shape = grad_x.shape();
    let dims = shape.dims();
    let mut result: Tensor<f32> = zeros(&dims)?;

    let total_elements = dims.iter().product::<usize>();

    for i in 0..total_elements {
        let indices = linear_to_indices(i, &dims);

        // Defensive bounds checking to prevent index out of bounds errors
        let mut valid_indices = true;
        for (idx, &dim_size) in indices.iter().zip(dims.iter()) {
            if *idx >= dim_size {
                valid_indices = false;
                break;
            }
        }

        if !valid_indices {
            continue; // Skip this invalid index
        }

        let gx: f32 = grad_x.get(&indices)?.clone().into();
        let gy: f32 = grad_y.get(&indices)?.clone().into();
        let magnitude = (gx * gx + gy * gy).sqrt();
        result.set(&indices, magnitude.into())?;
    }

    Ok(result)
}

fn compute_gradient_direction(grad_x: &Tensor<f32>, grad_y: &Tensor<f32>) -> Result<Tensor<f32>> {
    let shape = grad_x.shape();
    let dims = shape.dims();
    let mut result: Tensor<f32> = zeros(&dims)?;

    let total_elements = dims.iter().product::<usize>();

    for i in 0..total_elements {
        let indices = linear_to_indices(i, &dims);
        let gx: f32 = grad_x.get(&indices)?.clone().into();
        let gy: f32 = grad_y.get(&indices)?.clone().into();
        let direction = gy.atan2(gx);
        result.set(&indices, direction.into())?;
    }

    Ok(result)
}

fn non_maximum_suppression(
    magnitude: &Tensor<f32>,
    direction: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    let shape = magnitude.shape();
    let dims = shape.dims();

    if dims.len() != 3 {
        return Err(VisionError::InvalidShape("Expected 3D tensor".to_string()));
    }

    let (channels, height, width) = (dims[0], dims[1], dims[2]);
    let mut result: Tensor<f32> = zeros(&dims)?;

    for c in 0..channels {
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mag: f32 = magnitude.get(&[c, y, x])?.clone().into();
                let dir: f32 = direction.get(&[c, y, x])?.clone().into();

                // Convert direction to 0-180 degrees
                let angle = (dir * 180.0 / std::f32::consts::PI).abs() % 180.0;

                let (neighbor1, neighbor2) = if angle < 22.5 || angle >= 157.5 {
                    // Horizontal
                    (
                        magnitude.get(&[c, y, x - 1])?.clone().into(),
                        magnitude.get(&[c, y, x + 1])?.clone().into(),
                    )
                } else if angle >= 22.5 && angle < 67.5 {
                    // Diagonal /
                    (
                        magnitude.get(&[c, y - 1, x + 1])?.clone().into(),
                        magnitude.get(&[c, y + 1, x - 1])?.clone().into(),
                    )
                } else if angle >= 67.5 && angle < 112.5 {
                    // Vertical
                    (
                        magnitude.get(&[c, y - 1, x])?.clone().into(),
                        magnitude.get(&[c, y + 1, x])?.clone().into(),
                    )
                } else {
                    // Diagonal \
                    (
                        magnitude.get(&[c, y - 1, x - 1])?.clone().into(),
                        magnitude.get(&[c, y + 1, x + 1])?.clone().into(),
                    )
                };

                let suppressed_val = if mag >= neighbor1 && mag >= neighbor2 {
                    mag
                } else {
                    0.0
                };
                result.set(&[c, y, x], suppressed_val.into())?;
            }
        }
    }

    Ok(result)
}

fn double_threshold_and_hysteresis(
    edges: &Tensor<f32>,
    low_threshold: f32,
    high_threshold: f32,
) -> Result<Tensor<f32>> {
    let shape = edges.shape();
    let dims = shape.dims();
    let mut result: Tensor<f32> = zeros(&dims)?;

    if dims.len() != 3 {
        return Err(VisionError::InvalidShape("Expected 3D tensor".to_string()));
    }

    let (channels, height, width) = (dims[0], dims[1], dims[2]);

    // First pass: classify pixels
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let val: f32 = edges.get(&[c, y, x])?.clone().into();
                let classification = if val >= high_threshold {
                    255.0 // Strong edge
                } else if val >= low_threshold {
                    128.0 // Weak edge
                } else {
                    0.0 // Non-edge
                };
                result.set(&[c, y, x], classification.into())?;
            }
        }
    }

    // Second pass: hysteresis tracking
    let mut changed = true;
    while changed {
        changed = false;
        for c in 0..channels {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let val: f32 = result.get(&[c, y, x])?.clone().into();
                    if val == 128.0 {
                        // Weak edge
                        // Check if connected to strong edge
                        let mut has_strong_neighbor = false;
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if dy == 0 && dx == 0 {
                                    continue;
                                }
                                let neighbor_y = (y as i32 + dy) as usize;
                                let neighbor_x = (x as i32 + dx) as usize;
                                let neighbor_val: f32 =
                                    result.get(&[c, neighbor_y, neighbor_x])?.clone().into();
                                if neighbor_val == 255.0 {
                                    has_strong_neighbor = true;
                                    break;
                                }
                            }
                            if has_strong_neighbor {
                                break;
                            }
                        }

                        if has_strong_neighbor {
                            result.set(&[c, y, x], 255.0.into())?;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // Final pass: suppress remaining weak edges
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let val: f32 = result.get(&[c, y, x])?.clone().into();
                if val == 128.0 {
                    result.set(&[c, y, x], 0.0.into())?;
                }
            }
        }
    }

    Ok(result)
}

fn subtract_tensors(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    let shape = a.shape();
    let dims = shape.dims();
    let mut result: Tensor<f32> = zeros(&dims)?;

    let total_elements = dims.iter().product::<usize>();

    for i in 0..total_elements {
        let indices = linear_to_indices(i, &dims);
        let val_a: f32 = a.get(&indices)?.clone().into();
        let val_b: f32 = b.get(&indices)?.clone().into();
        result.set(&indices, (val_a - val_b).into())?;
    }

    Ok(result)
}

fn linear_to_indices(linear_index: usize, dims: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; dims.len()];
    let mut remaining = linear_index;

    for i in (0..dims.len()).rev() {
        let stride: usize = dims[i + 1..].iter().product();
        indices[i] = remaining / stride;
        remaining %= stride;
    }

    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_gaussian_blur() -> Result<()> {
        let image = zeros(&[3, 32, 32])?;
        let result = gaussian_blur(&image, 1.0)?;

        let result_shape = result.shape();
        assert_eq!(result_shape.dims(), &[3, 32, 32]);
        Ok(())
    }

    #[test]
    fn test_sobel_edge_detection() -> Result<()> {
        let image = zeros(&[1, 32, 32])?;
        let result = sobel_edge_detection(&image)?;

        let result_shape = result.shape();
        assert_eq!(result_shape.dims(), &[1, 32, 32]);
        Ok(())
    }

    #[test]
    fn test_morphological_operations() -> Result<()> {
        let image = zeros(&[1, 32, 32])?;

        let erosion_config = MorphologyConfig::erosion(3);
        let result = morphological_operation(&image, erosion_config)?;
        assert_eq!(result.shape().dims(), &[1, 32, 32]);

        let dilation_config = MorphologyConfig::dilation(3);
        let result = morphological_operation(&image, dilation_config)?;
        assert_eq!(result.shape().dims(), &[1, 32, 32]);

        Ok(())
    }

    #[test]
    fn test_median_filter() -> Result<()> {
        let image = zeros(&[3, 16, 16])?;
        let result = median_filter(&image, 3)?;

        assert_eq!(result.shape().dims(), &[3, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_bilateral_filter() -> Result<()> {
        let image = zeros(&[3, 16, 16])?;
        let result = bilateral_filter(&image, 2.0, 75.0, 5)?;

        assert_eq!(result.shape().dims(), &[3, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_edge_detection_configs() -> Result<()> {
        let sobel_config = EdgeDetectionConfig::sobel();
        assert_eq!(sobel_config.algorithm, EdgeDetectionAlgorithm::Sobel);

        let canny_config = EdgeDetectionConfig::canny(50.0, 150.0);
        assert_eq!(canny_config.algorithm, EdgeDetectionAlgorithm::Canny);
        assert_eq!(canny_config.low_threshold, 50.0);
        assert_eq!(canny_config.high_threshold, 150.0);

        Ok(())
    }

    #[test]
    fn test_gaussian_blur_config() -> Result<()> {
        let config = GaussianBlurConfig::with_sigma(2.0);
        assert_eq!(config.sigma, 2.0);
        // Kernel size should be odd and based on sigma
        assert!(config.kernel_size % 2 == 1);
        assert!(config.kernel_size >= 3);

        Ok(())
    }

    #[test]
    fn test_filtering_config_presets() -> Result<()> {
        let edge_config = FilteringConfig::edge_detection();
        assert_eq!(edge_config.padding, PaddingMode::Reflect);
        assert!(!edge_config.normalize_kernel);
        assert_eq!(edge_config.clamp_range, Some((0.0, 255.0)));

        let smooth_config = FilteringConfig::smoothing();
        assert_eq!(smooth_config.padding, PaddingMode::Reflect);
        assert!(smooth_config.normalize_kernel);
        assert_eq!(smooth_config.clamp_range, None);

        Ok(())
    }

    #[test]
    fn test_conv2d_basic() -> Result<()> {
        let image = zeros(&[1, 16, 16])?;
        let kernel = zeros(&[3, 3])?;
        let result = conv2d(&image, &kernel)?;

        assert_eq!(result.shape().dims(), &[1, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_invalid_inputs() -> Result<()> {
        // Test invalid kernel size for median filter
        let image = zeros(&[3, 16, 16])?;
        assert!(median_filter(&image, 4).is_err()); // Even kernel size should fail

        // Test morphological operations on multi-channel image
        assert!(morphological_operation(&image, MorphologyConfig::erosion(3)).is_err());

        // Test invalid convolution kernel
        let invalid_kernel = zeros(&[2, 4])?; // Even dimensions
        assert!(conv2d(&image, &invalid_kernel).is_err());

        Ok(())
    }
}
