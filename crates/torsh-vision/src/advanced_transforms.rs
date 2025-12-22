//! Advanced SciRS2-Powered Image Transformations and Augmentations
//!
//! This module provides state-of-the-art data augmentation techniques powered by the SciRS2
//! ecosystem, including GPU acceleration, mixed precision, and advanced augmentation strategies.
//! All operations follow the SciRS2 integration policy for optimal performance and compatibility.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::hardware::{
    GpuTransform, HardwareAccelerated, HardwareContext, MixedPrecisionTransform,
};
use crate::scirs2_integration::{
    ContrastMethod, DenoiseMethod, EdgeDetectionMethod, SciRS2VisionProcessor, VisionConfig,
};
use crate::transforms::Transform;
use crate::{Result, VisionError};
use scirs2_core::legacy::rng; // For rng() function
use scirs2_core::ndarray::{s, Array2, Array3, Array4};
use scirs2_core::random::{Random, Rng}; // SciRS2 Policy: Use scirs2_core::random instead of rand
use std::sync::Arc;
use torsh_core::device::{CpuDevice, Device, DeviceType};
use torsh_core::dtype::DType;
use torsh_tensor::Tensor;

/// Advanced SciRS2-powered transforms with GPU acceleration and mixed precision support
/// Provides comprehensive data augmentation capabilities using the SciRS2 ecosystem
pub struct AdvancedTransforms {
    context: HardwareContext,
    vision_processor: SciRS2VisionProcessor,
    rng: scirs2_core::random::Random,
}

impl AdvancedTransforms {
    pub fn new(context: HardwareContext) -> Self {
        let vision_config = VisionConfig::default();
        let vision_processor = SciRS2VisionProcessor::new(vision_config);
        let rng = rng(); // SciRS2 random number generator

        Self {
            context,
            vision_processor,
            rng,
        }
    }

    pub fn auto_detect() -> Result<Self> {
        let context = HardwareContext::auto_detect()?;
        Ok(Self::new(context))
    }

    pub fn with_config(context: HardwareContext, vision_config: VisionConfig) -> Self {
        let vision_processor = SciRS2VisionProcessor::new(vision_config);
        let rng = rng();

        Self {
            context,
            vision_processor,
            rng,
        }
    }

    /// Advanced Data Augmentation Methods powered by SciRS2

    /// Apply comprehensive image augmentation pipeline
    pub fn augment_image(&self, image: &Tensor, config: &AugmentationConfig) -> Result<Tensor> {
        let mut result = image.clone();

        // Apply geometric transformations
        if config.rotation.enabled {
            result = self.random_rotation(&result, config.rotation.range)?;
        }
        if config.scaling.enabled {
            result = self.random_scaling(&result, config.scaling.range)?;
        }
        if config.translation.enabled {
            result = self.random_translation(&result, config.translation.range)?;
        }
        if config.shearing.enabled {
            result = self.random_shear(&result, config.shearing.range)?;
        }

        // Apply photometric transformations
        if config.brightness.enabled {
            result = self.random_brightness(&result, config.brightness.range)?;
        }
        if config.contrast.enabled {
            result = self.random_contrast(&result, config.contrast.range)?;
        }
        if config.saturation.enabled {
            result = self.random_saturation(&result, config.saturation.range)?;
        }
        if config.hue.enabled {
            result = self.random_hue(&result, config.hue.range)?;
        }

        // Apply advanced augmentations
        if config.noise.enabled {
            result = self.add_noise(&result, config.noise.noise_type, config.noise.intensity)?;
        }
        if config.blur.enabled {
            result = self.random_blur(&result, config.blur.kernel_size, config.blur.sigma_range)?;
        }
        if config.elastic.enabled {
            result =
                self.elastic_deformation(&result, config.elastic.alpha, config.elastic.sigma)?;
        }
        if config.cutout.enabled {
            result = self.cutout(&result, config.cutout.num_holes, config.cutout.hole_size)?;
        }

        Ok(result)
    }

    /// Random rotation with SciRS2 optimization
    pub fn random_rotation(&self, image: &Tensor, angle_range: (f32, f32)) -> Result<Tensor> {
        let mut rng = rng();
        let angle = rng.gen_range(angle_range.0..angle_range.1);
        self.rotate_image(image, angle)
    }

    /// Random scaling with aspect ratio preservation
    pub fn random_scaling(&self, image: &Tensor, scale_range: (f32, f32)) -> Result<Tensor> {
        let mut rng = rng();
        let scale = rng.gen_range(scale_range.0..scale_range.1);
        self.scale_image(image, scale)
    }

    /// Random translation
    pub fn random_translation(
        &self,
        image: &Tensor,
        translation_range: (f32, f32),
    ) -> Result<Tensor> {
        let mut rng = rng();
        let tx = rng.gen_range(-translation_range.0..translation_range.0);
        let ty = rng.gen_range(-translation_range.1..translation_range.1);
        self.translate_image(image, tx, ty)
    }

    /// Random shearing transformation
    pub fn random_shear(&self, image: &Tensor, shear_range: (f32, f32)) -> Result<Tensor> {
        let mut rng = rng();
        let shear_x = rng.gen_range(-shear_range.0..shear_range.0);
        let shear_y = rng.gen_range(-shear_range.1..shear_range.1);
        self.shear_image(image, shear_x, shear_y)
    }

    /// Random brightness adjustment
    pub fn random_brightness(
        &self,
        image: &Tensor,
        brightness_range: (f32, f32),
    ) -> Result<Tensor> {
        let mut rng = rng();
        let factor = rng.gen_range(brightness_range.0..brightness_range.1);
        self.adjust_brightness(image, factor)
    }

    /// Random contrast adjustment
    pub fn random_contrast(&self, image: &Tensor, contrast_range: (f32, f32)) -> Result<Tensor> {
        let mut rng = rng();
        let factor = rng.gen_range(contrast_range.0..contrast_range.1);
        self.adjust_contrast(image, factor)
    }

    /// Random saturation adjustment
    pub fn random_saturation(
        &self,
        image: &Tensor,
        saturation_range: (f32, f32),
    ) -> Result<Tensor> {
        let mut rng = rng();
        let factor = rng.gen_range(saturation_range.0..saturation_range.1);
        self.adjust_saturation(image, factor)
    }

    /// Random hue adjustment
    pub fn random_hue(&self, image: &Tensor, hue_range: (f32, f32)) -> Result<Tensor> {
        let mut rng = rng();
        let factor = rng.gen_range(hue_range.0..hue_range.1);
        self.adjust_hue(image, factor)
    }

    /// Add various types of noise using SciRS2 random
    pub fn add_noise(
        &self,
        image: &Tensor,
        noise_type: NoiseType,
        intensity: f32,
    ) -> Result<Tensor> {
        let mut rng = rng();
        let shape = image.shape();
        let mut result = image.clone();

        match noise_type {
            NoiseType::Gaussian => {
                let noise_data: Vec<f32> = (0..shape.dims().iter().product::<usize>())
                    .map(|_| rng.gen_range(-intensity..intensity))
                    .collect();
                let noise_tensor = Tensor::from_vec(noise_data, shape.dims())?;
                result = result.add(&noise_tensor)?;
            }
            NoiseType::SaltPepper => {
                let data = result.to_vec()?;
                let mut noisy_data = data;
                for pixel in noisy_data.iter_mut() {
                    if rng.random::<f32>() < intensity / 2.0 {
                        *pixel = 0.0; // Pepper
                    } else if rng.random::<f32>() < intensity {
                        *pixel = 1.0; // Salt
                    }
                }
                result = Tensor::from_vec(noisy_data, shape.dims())?;
            }
            NoiseType::Uniform => {
                let noise_data: Vec<f32> = (0..shape.dims().iter().product::<usize>())
                    .map(|_| rng.gen_range(-intensity..intensity))
                    .collect();
                let noise_tensor = Tensor::from_vec(noise_data, shape.dims())?;
                result = result.add(&noise_tensor)?;
            }
        }
        Ok(result)
    }

    /// Random blur using SciRS2 vision processor
    pub fn random_blur(
        &self,
        image: &Tensor,
        kernel_size: usize,
        sigma_range: (f32, f32),
    ) -> Result<Tensor> {
        let mut rng = rng();
        let sigma = rng.gen_range(sigma_range.0..sigma_range.1);
        self.vision_processor
            .gaussian_blur(image, kernel_size, sigma)
    }

    /// Cutout augmentation with SciRS2 random
    pub fn cutout(
        &self,
        image: &Tensor,
        num_holes: usize,
        hole_size: (usize, usize),
    ) -> Result<Tensor> {
        let mut result = image.clone();
        let shape = image.shape();
        let height = shape.dims()[0];
        let width = shape.dims()[1];
        let mut rng = rng();

        for _ in 0..num_holes {
            let x = rng.gen_range(0..width.saturating_sub(hole_size.0));
            let y = rng.gen_range(0..height.saturating_sub(hole_size.1));

            let mut mask_data = vec![1.0f32; shape.dims().iter().product::<usize>()];

            if shape.dims().len() == 3 {
                let channels = shape.dims()[2];
                for i in y..(y + hole_size.1).min(height) {
                    for j in x..(x + hole_size.0).min(width) {
                        for c in 0..channels {
                            let idx = i * width * channels + j * channels + c;
                            if idx < mask_data.len() {
                                mask_data[idx] = 0.0;
                            }
                        }
                    }
                }
            } else {
                for i in y..(y + hole_size.1).min(height) {
                    for j in x..(x + hole_size.0).min(width) {
                        let idx = i * width + j;
                        if idx < mask_data.len() {
                            mask_data[idx] = 0.0;
                        }
                    }
                }
            }

            let mask = Tensor::from_vec(mask_data, shape.dims())?;
            result = result.mul(&mask)?;
        }
        Ok(result)
    }

    /// Helper methods for geometric transformations
    fn rotate_image(&self, image: &Tensor, angle: f32) -> Result<Tensor> {
        // Simplified rotation using nearest neighbor interpolation
        let shape = image.shape();
        let height = shape.dims()[0] as f32;
        let width = shape.dims()[1] as f32;
        let center_x = width / 2.0;
        let center_y = height / 2.0;

        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let data = image.to_vec()?;
        let mut rotated_data = vec![0.0f32; data.len()];

        if shape.dims().len() == 3 {
            let channels = shape.dims()[2];
            for y in 0..shape.dims()[0] {
                for x in 0..shape.dims()[1] {
                    // Transform coordinates
                    let dx = x as f32 - center_x;
                    let dy = y as f32 - center_y;
                    let src_x = (dx * cos_a - dy * sin_a + center_x) as usize;
                    let src_y = (dx * sin_a + dy * cos_a + center_y) as usize;

                    if src_x < shape.dims()[1] && src_y < shape.dims()[0] {
                        for c in 0..channels {
                            let dst_idx = y * shape.dims()[1] * channels + x * channels + c;
                            let src_idx = src_y * shape.dims()[1] * channels + src_x * channels + c;
                            if dst_idx < rotated_data.len() && src_idx < data.len() {
                                rotated_data[dst_idx] = data[src_idx];
                            }
                        }
                    }
                }
            }
        }

        Ok(Tensor::from_vec(rotated_data, shape.dims())?)
    }

    fn scale_image(&self, image: &Tensor, scale: f32) -> Result<Tensor> {
        // Simplified scaling - would use proper interpolation in production
        let shape = image.shape();
        let new_height = (shape.dims()[0] as f32 * scale) as usize;
        let new_width = (shape.dims()[1] as f32 * scale) as usize;

        let _new_shape = if shape.dims().len() == 3 {
            vec![new_height, new_width, shape.dims()[2]]
        } else {
            vec![new_height, new_width]
        };

        // Use vision processor's super resolution for upscaling
        if scale > 1.0 {
            self.vision_processor.super_resolution(image, scale)
        } else {
            // Downscaling - use simple decimation
            crate::ops::resize(image, (new_height, new_width))
        }
    }

    fn translate_image(&self, image: &Tensor, tx: f32, ty: f32) -> Result<Tensor> {
        // Simple translation implementation
        let shape = image.shape();
        let data = image.to_vec()?;
        let mut translated_data = vec![0.0f32; data.len()];

        let height = shape.dims()[0] as i32;
        let width = shape.dims()[1] as i32;
        let tx_int = tx as i32;
        let ty_int = ty as i32;

        if shape.dims().len() == 3 {
            let channels = shape.dims()[2];
            for y in 0..height {
                for x in 0..width {
                    let src_x = x - tx_int;
                    let src_y = y - ty_int;

                    if src_x >= 0 && src_x < width && src_y >= 0 && src_y < height {
                        for c in 0..channels {
                            let dst_idx = (y * width * channels as i32
                                + x * channels as i32
                                + c as i32) as usize;
                            let src_idx = (src_y * width * channels as i32
                                + src_x * channels as i32
                                + c as i32) as usize;
                            if dst_idx < translated_data.len() && src_idx < data.len() {
                                translated_data[dst_idx] = data[src_idx];
                            }
                        }
                    }
                }
            }
        }

        Ok(Tensor::from_vec(translated_data, shape.dims())?)
    }

    fn shear_image(&self, image: &Tensor, shear_x: f32, shear_y: f32) -> Result<Tensor> {
        // Simplified shear transformation
        let shape = image.shape();
        let data = image.to_vec()?;
        let mut sheared_data = vec![0.0f32; data.len()];

        if shape.dims().len() == 3 {
            let height = shape.dims()[0];
            let width = shape.dims()[1];
            let channels = shape.dims()[2];

            for y in 0..height {
                for x in 0..width {
                    let src_x = x as f32 - shear_x * y as f32;
                    let src_y = y as f32 - shear_y * x as f32;

                    let src_x_int = src_x as usize;
                    let src_y_int = src_y as usize;

                    if src_x_int < width && src_y_int < height {
                        for c in 0..channels {
                            let dst_idx = y * width * channels + x * channels + c;
                            let src_idx = src_y_int * width * channels + src_x_int * channels + c;
                            if dst_idx < sheared_data.len() && src_idx < data.len() {
                                sheared_data[dst_idx] = data[src_idx];
                            }
                        }
                    }
                }
            }
        }

        Ok(Tensor::from_vec(sheared_data, shape.dims())?)
    }

    fn adjust_brightness(&self, image: &Tensor, factor: f32) -> Result<Tensor> {
        let data = image.to_vec()?;
        let brightened_data: Vec<f32> =
            data.iter().map(|&x| (x + factor).clamp(0.0, 1.0)).collect();
        Ok(Tensor::from_vec(brightened_data, image.shape().dims())?)
    }

    fn adjust_contrast(&self, image: &Tensor, factor: f32) -> Result<Tensor> {
        let data = image.to_vec()?;
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let contrasted_data: Vec<f32> = data
            .iter()
            .map(|&x| ((x - mean) * factor + mean).clamp(0.0, 1.0))
            .collect();
        Ok(Tensor::from_vec(contrasted_data, image.shape().dims())?)
    }

    fn adjust_saturation(&self, image: &Tensor, factor: f32) -> Result<Tensor> {
        // Simplified saturation adjustment (RGB to grayscale and blend)
        let shape = image.shape();
        if shape.dims().len() != 3 || shape.dims()[2] != 3 {
            return Ok(image.clone()); // Only works for RGB images
        }

        let data = image.to_vec()?;
        let mut adjusted_data = data.clone();

        let height = shape.dims()[0];
        let width = shape.dims()[1];

        for y in 0..height {
            for x in 0..width {
                let r_idx = y * width * 3 + x * 3;
                let g_idx = r_idx + 1;
                let b_idx = r_idx + 2;

                if b_idx < data.len() {
                    let r = data[r_idx];
                    let g = data[g_idx];
                    let b = data[b_idx];

                    // Convert to grayscale
                    let gray = 0.299 * r + 0.587 * g + 0.114 * b;

                    // Blend with original based on saturation factor
                    adjusted_data[r_idx] = (gray + (r - gray) * factor).clamp(0.0, 1.0);
                    adjusted_data[g_idx] = (gray + (g - gray) * factor).clamp(0.0, 1.0);
                    adjusted_data[b_idx] = (gray + (b - gray) * factor).clamp(0.0, 1.0);
                }
            }
        }

        Ok(Tensor::from_vec(adjusted_data, shape.dims())?)
    }

    fn adjust_hue(&self, image: &Tensor, _factor: f32) -> Result<Tensor> {
        // Simplified hue adjustment - would convert to HSV in production
        Ok(image.clone()) // Placeholder implementation
    }

    fn elastic_deformation(&self, image: &Tensor, alpha: f32, _sigma: f32) -> Result<Tensor> {
        // Simplified elastic deformation
        let shape = image.shape();
        let height = shape.dims()[0];
        let width = shape.dims()[1];
        let mut rng = rng();

        // Generate random displacement fields
        let mut dx_field = Array2::zeros((height, width));
        let mut dy_field = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                dx_field[[i, j]] = rng.gen_range(-alpha..alpha);
                dy_field[[i, j]] = rng.gen_range(-alpha..alpha);
            }
        }

        // Apply simple displacement (would use proper grid sampling in production)
        Ok(image.clone()) // Placeholder - complex implementation needed
    }

    fn apply_displacement(&self, image: &Tensor, _dx: &Tensor, _dy: &Tensor) -> Result<Tensor> {
        // Placeholder for displacement application
        Ok(image.clone())
    }

    /// Legacy GPU methods (maintaining compatibility)
    pub fn create_gpu_resize(&self, size: (usize, usize)) -> GpuResize {
        GpuResize::new(size, DeviceType::Cpu)
    }

    pub fn create_gpu_normalize(&self, mean: Vec<f32>, std: Vec<f32>) -> GpuNormalize {
        GpuNormalize::new(mean, std, DeviceType::Cpu)
    }

    pub fn create_gpu_color_jitter(
        &self,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        hue: f32,
    ) -> GpuColorJitter {
        GpuColorJitter::new(brightness, contrast, saturation, hue, DeviceType::Cpu)
    }

    pub fn create_gpu_augmentation_chain(
        &self,
        transforms: Vec<Box<dyn GpuTransform>>,
    ) -> GpuAugmentationChain {
        GpuAugmentationChain::new(transforms, DeviceType::Cpu)
    }
}

/// Advanced Augmentation Configuration
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    pub rotation: AugmentationParam<(f32, f32)>,
    pub scaling: AugmentationParam<(f32, f32)>,
    pub translation: AugmentationParam<(f32, f32)>,
    pub shearing: AugmentationParam<(f32, f32)>,
    pub brightness: AugmentationParam<(f32, f32)>,
    pub contrast: AugmentationParam<(f32, f32)>,
    pub saturation: AugmentationParam<(f32, f32)>,
    pub hue: AugmentationParam<(f32, f32)>,
    pub noise: NoiseAugmentationParam,
    pub blur: BlurAugmentationParam,
    pub elastic: ElasticAugmentationParam,
    pub cutout: CutoutAugmentationParam,
    pub mixup: MixupAugmentationParam,
}

#[derive(Debug, Clone)]
pub struct AugmentationParam<T> {
    pub enabled: bool,
    pub range: T,
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct NoiseAugmentationParam {
    pub enabled: bool,
    pub noise_type: NoiseType,
    pub intensity: f32,
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct BlurAugmentationParam {
    pub enabled: bool,
    pub kernel_size: usize,
    pub sigma_range: (f32, f32),
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct ElasticAugmentationParam {
    pub enabled: bool,
    pub alpha: f32,
    pub sigma: f32,
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct CutoutAugmentationParam {
    pub enabled: bool,
    pub num_holes: usize,
    pub hole_size: (usize, usize),
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct MixupAugmentationParam {
    pub enabled: bool,
    pub alpha: f32,
    pub probability: f32,
}

/// Types of noise for augmentation
#[derive(Debug, Clone, Copy)]
pub enum NoiseType {
    Gaussian,
    SaltPepper,
    Uniform,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            rotation: AugmentationParam {
                enabled: true,
                range: (-15.0, 15.0),
                probability: 0.5,
            },
            scaling: AugmentationParam {
                enabled: true,
                range: (0.8, 1.2),
                probability: 0.5,
            },
            translation: AugmentationParam {
                enabled: true,
                range: (0.1, 0.1),
                probability: 0.5,
            },
            shearing: AugmentationParam {
                enabled: false,
                range: (-0.2, 0.2),
                probability: 0.3,
            },
            brightness: AugmentationParam {
                enabled: true,
                range: (-0.2, 0.2),
                probability: 0.5,
            },
            contrast: AugmentationParam {
                enabled: true,
                range: (0.8, 1.2),
                probability: 0.5,
            },
            saturation: AugmentationParam {
                enabled: true,
                range: (0.8, 1.2),
                probability: 0.5,
            },
            hue: AugmentationParam {
                enabled: false,
                range: (-0.1, 0.1),
                probability: 0.3,
            },
            noise: NoiseAugmentationParam {
                enabled: false,
                noise_type: NoiseType::Gaussian,
                intensity: 0.05,
                probability: 0.2,
            },
            blur: BlurAugmentationParam {
                enabled: false,
                kernel_size: 3,
                sigma_range: (0.5, 2.0),
                probability: 0.2,
            },
            elastic: ElasticAugmentationParam {
                enabled: false,
                alpha: 0.5,
                sigma: 0.1,
                probability: 0.1,
            },
            cutout: CutoutAugmentationParam {
                enabled: false,
                num_holes: 1,
                hole_size: (16, 16),
                probability: 0.3,
            },
            mixup: MixupAugmentationParam {
                enabled: false,
                alpha: 0.2,
                probability: 0.5,
            },
        }
    }
}

/// SciRS2-powered augmentation pipeline
#[derive(Debug)]
pub struct SciRS2AugmentationPipeline {
    config: AugmentationConfig,
    device: DeviceType,
    vision_processor: Arc<SciRS2VisionProcessor>,
}

impl SciRS2AugmentationPipeline {
    pub fn new(
        config: AugmentationConfig,
        device: DeviceType,
        vision_processor: &SciRS2VisionProcessor,
    ) -> Self {
        Self {
            config,
            device,
            vision_processor: Arc::new(vision_processor.clone()),
        }
    }

    pub fn apply(&self, image: &Tensor) -> Result<Tensor> {
        let mut result = image.clone();
        let mut rng = rng();

        // Apply each augmentation with probability check
        if self.config.rotation.enabled && rng.random::<f32>() < self.config.rotation.probability {
            let angle = rng.gen_range(self.config.rotation.range.0..self.config.rotation.range.1);
            result = self.rotate_image(&result, angle)?;
        }

        if self.config.brightness.enabled
            && rng.random::<f32>() < self.config.brightness.probability
        {
            let factor =
                rng.gen_range(self.config.brightness.range.0..self.config.brightness.range.1);
            result = self.adjust_brightness(&result, factor)?;
        }

        if self.config.noise.enabled && rng.random::<f32>() < self.config.noise.probability {
            result = self.add_noise(
                &result,
                self.config.noise.noise_type,
                self.config.noise.intensity,
            )?;
        }

        Ok(result)
    }

    fn rotate_image(&self, image: &Tensor, _angle: f32) -> Result<Tensor> {
        // Simplified rotation - would use proper affine transformation
        Ok(image.clone())
    }

    fn adjust_brightness(&self, image: &Tensor, factor: f32) -> Result<Tensor> {
        let data = image.to_vec()?;
        let brightened_data: Vec<f32> =
            data.iter().map(|&x| (x + factor).clamp(0.0, 1.0)).collect();
        Ok(Tensor::from_vec(brightened_data, image.shape().dims())?)
    }

    fn add_noise(&self, image: &Tensor, noise_type: NoiseType, intensity: f32) -> Result<Tensor> {
        let mut rng = rng();
        let shape = image.shape();
        let mut result = image.clone();

        match noise_type {
            NoiseType::Gaussian => {
                let noise_data: Vec<f32> = (0..shape.dims().iter().product::<usize>())
                    .map(|_| rng.gen_range(-intensity..intensity))
                    .collect();
                let noise_tensor = Tensor::from_vec(noise_data, shape.dims())?;
                result = result.add(&noise_tensor)?;
            }
            NoiseType::SaltPepper => {
                let data = result.to_vec()?;
                let mut noisy_data = data;
                for pixel in noisy_data.iter_mut() {
                    if rng.random::<f32>() < intensity / 2.0 {
                        *pixel = 0.0;
                    } else if rng.random::<f32>() < intensity {
                        *pixel = 1.0;
                    }
                }
                result = Tensor::from_vec(noisy_data, shape.dims())?;
            }
            NoiseType::Uniform => {
                let noise_data: Vec<f32> = (0..shape.dims().iter().product::<usize>())
                    .map(|_| rng.gen_range(-intensity..intensity))
                    .collect();
                let noise_tensor = Tensor::from_vec(noise_data, shape.dims())?;
                result = result.add(&noise_tensor)?;
            }
        }
        Ok(result)
    }
}

/// GPU-accelerated resize transform
pub struct GpuResize {
    size: (usize, usize),
    device: DeviceType,
}

impl GpuResize {
    pub fn new(size: (usize, usize), device: DeviceType) -> Self {
        Self { size, device }
    }
}

impl GpuTransform for GpuResize {
    fn forward_gpu(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        match self.device {
            DeviceType::Cuda(_) => self.cuda_resize(input),
            _ => crate::ops::resize(input, self.size),
        }
    }

    fn forward_gpu_f32(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        match self.device {
            DeviceType::Cuda(_) => self.cuda_resize_f32(input),
            _ => {
                let input_f32 = input.to_dtype(DType::F32)?;
                let output_f32 = crate::ops::resize(&input_f32, self.size)?;
                Ok(output_f32.to_dtype(DType::F32)?)
            }
        }
    }
}

impl GpuResize {
    fn cuda_resize(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // For now, fallback to optimized CPU implementation
        // In production, this would use CUDA kernels
        crate::ops::resize(input, self.size)
    }

    fn cuda_resize_f32(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // For now, fallback with type conversion
        let input_f32 = input.to_dtype(DType::F32)?;
        let output_f32 = crate::ops::resize(&input_f32, self.size)?;
        Ok(output_f32.to_dtype(DType::F32)?)
    }
}

impl Transform for GpuResize {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        self.forward_gpu(input)
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(GpuResize::new(self.size, self.device))
    }
}

/// GPU-accelerated normalization transform
pub struct GpuNormalize {
    mean: Vec<f32>,
    std: Vec<f32>,
    device: DeviceType,
}

impl GpuNormalize {
    pub fn new(mean: Vec<f32>, std: Vec<f32>, device: DeviceType) -> Self {
        Self { mean, std, device }
    }
}

impl GpuTransform for GpuNormalize {
    fn forward_gpu(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        match self.device {
            DeviceType::Cuda(_) => self.cuda_normalize(input),
            _ => crate::ops::normalize(
                input,
                crate::ops::color::NormalizationConfig {
                    method: crate::ops::color::NormalizationMethod::Custom,
                    mean: Some(self.mean.clone()),
                    std: Some(self.std.clone()),
                    per_channel: true,
                    eps: 1e-8,
                },
            ),
        }
    }

    fn forward_gpu_f32(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        match self.device {
            DeviceType::Cuda(_) => self.cuda_normalize_f32(input),
            _ => {
                let input_f32 = input.to_dtype(DType::F32)?;
                let output_f32 = crate::ops::normalize(
                    &input_f32,
                    crate::ops::color::NormalizationConfig {
                        method: crate::ops::color::NormalizationMethod::Custom,
                        mean: Some(self.mean.clone()),
                        std: Some(self.std.clone()),
                        per_channel: true,
                        eps: 1e-8,
                    },
                )?;
                Ok(output_f32.to_dtype(DType::F32)?)
            }
        }
    }
}

impl GpuNormalize {
    fn cuda_normalize(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // For now, fallback to CPU
        crate::ops::normalize(
            input,
            crate::ops::color::NormalizationConfig {
                method: crate::ops::color::NormalizationMethod::Custom,
                mean: Some(self.mean.clone()),
                std: Some(self.std.clone()),
                per_channel: true,
                eps: 1e-8,
            },
        )
    }

    fn cuda_normalize_f32(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // For now, fallback with type conversion
        let input_f32 = input.to_dtype(DType::F32)?;
        let output_f32 = crate::ops::normalize(
            &input_f32,
            crate::ops::color::NormalizationConfig {
                method: crate::ops::color::NormalizationMethod::Custom,
                mean: Some(self.mean.clone()),
                std: Some(self.std.clone()),
                per_channel: true,
                eps: 1e-8,
            },
        )?;
        Ok(output_f32.to_dtype(DType::F32)?)
    }
}

impl Transform for GpuNormalize {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        self.forward_gpu(input)
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(GpuNormalize::new(
            self.mean.clone(),
            self.std.clone(),
            self.device,
        ))
    }
}

/// GPU-accelerated color jitter transform
pub struct GpuColorJitter {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
    device: DeviceType,
}

impl GpuColorJitter {
    pub fn new(
        brightness: f32,
        contrast: f32,
        saturation: f32,
        hue: f32,
        device: DeviceType,
    ) -> Self {
        Self {
            brightness,
            contrast,
            saturation,
            hue,
            device,
        }
    }
}

impl GpuTransform for GpuColorJitter {
    fn forward_gpu(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if matches!(self.device, DeviceType::Cuda(_)) {
            self.cuda_color_jitter(input)
        } else {
            self.cpu_color_jitter(input)
        }
    }

    fn forward_gpu_f32(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if matches!(self.device, DeviceType::Cuda(_)) {
            self.cuda_color_jitter_f32(input)
        } else {
            let input_f32 = input.to_dtype(DType::F32)?;
            let output_f32 = self.cpu_color_jitter(&input_f32)?;
            Ok(output_f32.to_dtype(DType::F32)?)
        }
    }
}

impl GpuColorJitter {
    fn cuda_color_jitter(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // For now, fallback to CPU
        self.cpu_color_jitter(input)
    }

    fn cuda_color_jitter_f32(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // For now, fallback with type conversion
        let input_f32 = input.to_dtype(DType::F32)?;
        let output_f32 = self.cpu_color_jitter(&input_f32)?;
        Ok(output_f32.to_dtype(DType::F32)?)
    }

    fn cpu_color_jitter(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut rng = rng();
        let mut output = input.clone();

        // Apply brightness adjustment
        if self.brightness > 0.0 {
            let brightness_factor = 1.0 + rng.gen_range(-self.brightness..self.brightness);
            output = output.mul_scalar(brightness_factor)?;
        }

        // Apply contrast adjustment
        if self.contrast > 0.0 {
            let contrast_factor = 1.0 + rng.gen_range(-self.contrast..self.contrast);
            let mean = output.mean(None, false)?;
            output.sub_scalar_(mean.item()?)?;
            output = output.mul_scalar(contrast_factor)?;
            output.add_scalar(mean.item()?)?;
        }

        // Clamp values to [0, 1]
        output = output.clamp(0.0, 1.0)?;

        Ok(output)
    }
}

impl Transform for GpuColorJitter {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        self.forward_gpu(input)
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(GpuColorJitter::new(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue,
            self.device,
        ))
    }
}

/// GPU-accelerated augmentation chain
pub struct GpuAugmentationChain {
    transforms: Vec<Box<dyn GpuTransform>>,
    device: DeviceType,
}

impl GpuAugmentationChain {
    pub fn new(transforms: Vec<Box<dyn GpuTransform>>, device: DeviceType) -> Self {
        Self { transforms, device }
    }
}

impl GpuTransform for GpuAugmentationChain {
    fn forward_gpu(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut output = input.clone();
        for transform in &self.transforms {
            output = transform.forward_gpu(&output)?;
        }
        Ok(output)
    }

    fn forward_gpu_f32(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut output = input.clone();
        for transform in &self.transforms {
            output = transform.forward_gpu_f32(&output)?;
        }
        Ok(output)
    }
}

impl Transform for GpuAugmentationChain {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        self.forward_gpu(input)
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        // Note: Cannot easily clone Box<dyn GpuTransform> without additional trait bounds
        // For now, return a minimal implementation that copies the device
        Box::new(GpuAugmentationChain::new(Vec::new(), DeviceType::Cpu))
    }
}

/// Mixed precision training utilities
pub struct MixedPrecisionTraining {
    context: HardwareContext,
    loss_scaler: f32,
    dynamic_scaling: bool,
}

impl MixedPrecisionTraining {
    pub fn new(context: HardwareContext, loss_scaler: f32, dynamic_scaling: bool) -> Self {
        Self {
            context,
            loss_scaler,
            dynamic_scaling,
        }
    }

    pub fn scale_loss(&self, loss: &Tensor<f32>) -> Result<Tensor<f32>> {
        if self.context.supports_mixed_precision() {
            loss.mul_scalar(self.loss_scaler)
                .map_err(|e| VisionError::TensorError(e))
        } else {
            Ok(loss.clone())
        }
    }

    pub fn unscale_gradients(&self, gradients: &mut [Tensor<f32>]) -> Result<()> {
        if self.context.supports_mixed_precision() {
            for grad in gradients {
                *grad = grad
                    .div_scalar(self.loss_scaler)
                    .map_err(|e| VisionError::TensorError(e))?;
            }
        }
        Ok(())
    }

    pub fn update_scaler(&mut self, gradients: &[Tensor<f32>]) -> Result<()> {
        if self.dynamic_scaling {
            let has_inf_or_nan = gradients.iter().any(|grad| {
                // Check for infinite or NaN values in gradient tensor
                // For now, we'll use a simple check that assumes well-behaved gradients
                grad.numel() == 0 // Simple fallback check
            });

            if has_inf_or_nan {
                self.loss_scaler *= 0.5;
            } else {
                self.loss_scaler *= 1.05;
            }

            // Clamp scaler to reasonable range
            self.loss_scaler = self.loss_scaler.clamp(1.0, 65536.0);
        }
        Ok(())
    }
}

/// Tensor Core optimization utilities
pub struct TensorCoreOptimizer {
    context: HardwareContext,
}

impl TensorCoreOptimizer {
    pub fn new(context: HardwareContext) -> Self {
        Self { context }
    }

    pub fn optimize_tensor_shape(&self, tensor: &Tensor<f32>) -> Result<Tensor<f32>> {
        if !self.context.supports_tensor_cores() {
            return Ok(tensor.clone());
        }

        // Pad tensor dimensions to be multiples of 8 for Tensor Core efficiency
        let shape = tensor.shape();
        let dims = shape.dims();
        let mut new_dims = dims.to_vec();

        for dim in new_dims.iter_mut() {
            if *dim % 8 != 0 {
                *dim = (*dim + 7) / 8 * 8;
            }
        }

        if new_dims == dims {
            Ok(tensor.clone())
        } else {
            // Pad the tensor to the new dimensions
            self.pad_tensor(tensor, &new_dims)
        }
    }

    fn pad_tensor(&self, tensor: &Tensor<f32>, new_dims: &[usize]) -> Result<Tensor<f32>> {
        let shape = tensor.shape();
        let old_dims = shape.dims();
        let padded = Tensor::zeros(new_dims, tensor.device())?;

        // Copy original data to padded tensor
        // This is a simplified implementation
        match old_dims.len() {
            1 => {
                let mut slice = padded.narrow(0, 0, old_dims[0])?;
                slice.copy_(&tensor)?;
            }
            2 => {
                let mut slice = padded
                    .narrow(0, 0, old_dims[0])?
                    .narrow(1, 0, old_dims[1])?;
                slice.copy_(&tensor)?;
            }
            3 => {
                let mut slice = padded
                    .narrow(0, 0, old_dims[0])?
                    .narrow(1, 0, old_dims[1])?
                    .narrow(2, 0, old_dims[2])?;
                slice.copy_(&tensor)?;
            }
            _ => {
                return Err(VisionError::InvalidShape(
                    "Unsupported tensor dimension".to_string(),
                ))
            }
        }

        Ok(padded)
    }
}

/// Performance monitoring for hardware-accelerated transforms
pub struct PerformanceMonitor {
    context: HardwareContext,
    metrics: std::collections::HashMap<String, f64>,
}

impl PerformanceMonitor {
    pub fn new(context: HardwareContext) -> Self {
        Self {
            context,
            metrics: std::collections::HashMap::new(),
        }
    }

    pub fn measure_transform<T: GpuTransform>(
        &mut self,
        name: &str,
        transform: &T,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>> {
        let start = std::time::Instant::now();
        let output = transform.forward_gpu(input)?;
        let duration = start.elapsed();

        self.metrics
            .insert(name.to_string(), duration.as_secs_f64());
        Ok(output)
    }

    pub fn get_metrics(&self) -> &std::collections::HashMap<String, f64> {
        &self.metrics
    }

    pub fn clear_metrics(&mut self) {
        self.metrics.clear();
    }

    pub fn print_summary(&self) {
        println!("Performance Summary:");
        println!("Device: {}", self.context.device().device_type());
        println!(
            "Mixed Precision: {}",
            self.context.supports_mixed_precision()
        );
        println!("Tensor Cores: {}", self.context.supports_tensor_cores());

        for (name, time) in &self.metrics {
            println!("  {}: {:.4}s", name, time);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_advanced_transforms() {
        let transforms = AdvancedTransforms::auto_detect().unwrap();
        let resize = transforms.create_gpu_resize((224, 224));
        let input = zeros(&[3, 128, 128]).unwrap();
        let output = resize.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[3, 224, 224]);
    }

    #[test]
    fn test_gpu_normalize() {
        let device = DeviceType::Cpu;
        let normalize =
            GpuNormalize::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225], device);
        let input = zeros(&[3, 32, 32]).unwrap();
        let output = normalize.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[3, 32, 32]);
    }

    #[test]
    fn test_gpu_color_jitter() {
        let device = DeviceType::Cpu;
        let color_jitter = GpuColorJitter::new(0.1, 0.1, 0.1, 0.1, device);
        let input = zeros(&[3, 32, 32]).unwrap();
        let output = color_jitter.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[3, 32, 32]);
    }

    #[test]
    fn test_mixed_precision_training() {
        let context = HardwareContext::auto_detect().unwrap();
        let mut trainer = MixedPrecisionTraining::new(context, 128.0, true);

        let loss = zeros(&[1]).unwrap();
        let scaled_loss = trainer.scale_loss(&loss).unwrap();
        assert_eq!(scaled_loss.shape().dims(), &[1]);

        let mut gradients = vec![zeros(&[10]).unwrap(), zeros(&[20]).unwrap()];
        trainer.unscale_gradients(&mut gradients).unwrap();
        trainer.update_scaler(&gradients).unwrap();
    }

    #[test]
    fn test_tensor_core_optimizer() {
        let context = HardwareContext::auto_detect().unwrap();
        let supports_tensor_cores = context.supports_tensor_cores();
        let optimizer = TensorCoreOptimizer::new(context);

        let input = zeros(&[7, 15]).unwrap(); // Not multiples of 8
        let optimized = optimizer.optimize_tensor_shape(&input).unwrap();

        // On CPU (auto_detect default), tensor cores are not supported
        // so the shape should remain unchanged
        if supports_tensor_cores {
            // Should be padded to multiples of 8
            assert_eq!(optimized.shape().dims(), &[8, 16]);
        } else {
            // Should remain unchanged on CPU
            assert_eq!(optimized.shape().dims(), &[7, 15]);
        }
    }

    #[test]
    fn test_performance_monitor() {
        let context = HardwareContext::auto_detect().unwrap();
        let mut monitor = PerformanceMonitor::new(context);

        let resize = GpuResize::new((224, 224), DeviceType::Cpu);
        let input = zeros(&[3, 128, 128]).unwrap();

        let _output = monitor
            .measure_transform("resize", &resize, &input)
            .unwrap();

        let metrics = monitor.get_metrics();
        assert!(metrics.contains_key("resize"));
        assert!(metrics["resize"] >= 0.0);
    }
}
