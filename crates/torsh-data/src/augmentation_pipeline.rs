//! Data augmentation pipeline for machine learning training
//!
//! This module provides a comprehensive augmentation system for training data preprocessing.
//! Augmentation is a critical technique for improving model generalization by applying
//! random transformations to training data.
//!
//! # Features
//!
//! - **Pipeline composition**: AugmentationPipeline for chaining multiple transforms
//! - **Probabilistic transforms**: ConditionalTransform for random application
//! - **Image augmentations**: Color, brightness, contrast, and geometric transforms
//! - **Noise augmentation**: GaussianNoise for regularization
//! - **Cutout/Erasing**: RandomErasing for occlusion robustness
//! - **Preset pipelines**: Common augmentation configurations for different use cases

use crate::transforms::Transform;
use torsh_core::dtype::{FloatElement, TensorElement};
use torsh_core::error::Result;
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

#[cfg(feature = "std")]
use scirs2_core::random::thread_rng;

#[cfg(not(feature = "std"))]
use scirs2_core::random::thread_rng;

/// Augmentation pipeline builder for easy composition of transforms
pub struct AugmentationPipeline<T> {
    transforms: Vec<Box<dyn Transform<T, Output = T> + Send + Sync>>,
    probability: f32,
}

impl<T: 'static + Send + Sync> AugmentationPipeline<T> {
    /// Create a new augmentation pipeline
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            probability: 1.0,
        }
    }

    /// Set the probability of applying the entire pipeline
    pub fn with_probability(mut self, prob: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability must be between 0 and 1"
        );
        self.probability = prob;
        self
    }

    /// Add a transform to the pipeline
    pub fn add_transform<F>(mut self, transform: F) -> Self
    where
        F: Transform<T, Output = T> + 'static,
    {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Add a conditional transform that only applies with given probability
    pub fn add_conditional<F>(self, transform: F, prob: f32) -> Self
    where
        F: Transform<T, Output = T> + 'static,
    {
        self.add_transform(ConditionalTransform::new(transform, prob))
    }
}

impl<T: 'static + Send + Sync> Default for AugmentationPipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for AugmentationPipeline<T> {
    type Output = T;

    fn transform(&self, mut input: T) -> Result<Self::Output> {
        let mut rng = thread_rng();

        // Check if we should apply the pipeline at all
        if rng.random::<f32>() > self.probability {
            return Ok(input);
        }

        // Apply all transforms in sequence
        for transform in &self.transforms {
            input = transform.transform(input)?;
        }

        Ok(input)
    }
}

/// Conditional transform that applies with a given probability
pub struct ConditionalTransform<T, F> {
    transform: F,
    probability: f32,
    _phantom: core::marker::PhantomData<T>,
}

impl<T, F> ConditionalTransform<T, F> {
    pub fn new(transform: F, probability: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be between 0 and 1"
        );
        Self {
            transform,
            probability,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<T, F> Transform<T> for ConditionalTransform<T, F>
where
    F: Transform<T, Output = T>,
    T: Send + Sync,
{
    type Output = T;

    fn transform(&self, input: T) -> Result<Self::Output> {
        let mut rng = thread_rng();

        if rng.random::<f32>() < self.probability {
            self.transform.transform(input)
        } else {
            Ok(input)
        }
    }
}

/// Random brightness adjustment
pub struct RandomBrightness {
    factor_range: (f32, f32),
}

impl RandomBrightness {
    pub fn new(factor_range: (f32, f32)) -> Self {
        assert!(factor_range.0 <= factor_range.1, "Invalid factor range");
        Self { factor_range }
    }

    /// Create with symmetric range around 1.0
    pub fn symmetric(factor: f32) -> Self {
        Self::new((1.0 - factor, 1.0 + factor))
    }
}

impl<T: FloatElement> Transform<Tensor<T>> for RandomBrightness {
    type Output = Tensor<T>;

    fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
        let mut rng = thread_rng();
        let (lo, hi) = self.factor_range;
        let factor = lo + rng.random::<f32>() * (hi - lo);
        let shape = input.shape().dims().to_vec();
        let device = input.device();
        let data = input
            .to_vec()
            .map_err(|e| torsh_core::error::TorshError::Other(format!("to_vec failed: {}", e)))?;
        let out: Vec<T> = data
            .iter()
            .map(|&x| {
                let v = <T as TensorElement>::to_f64(&x).unwrap_or(0.0) * factor as f64;
                let clamped = v.max(0.0).min(1.0);
                <T as TensorElement>::from_f64(clamped).unwrap_or(x)
            })
            .collect();
        Tensor::from_data(out, shape, device).map_err(|e| e.into())
    }
}

/// Random contrast adjustment
pub struct RandomContrast {
    factor_range: (f32, f32),
}

impl RandomContrast {
    pub fn new(factor_range: (f32, f32)) -> Self {
        assert!(factor_range.0 <= factor_range.1, "Invalid factor range");
        Self { factor_range }
    }

    /// Create with symmetric range around 1.0
    pub fn symmetric(factor: f32) -> Self {
        Self::new((1.0 - factor, 1.0 + factor))
    }
}

impl<T: FloatElement> Transform<Tensor<T>> for RandomContrast {
    type Output = Tensor<T>;

    fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
        let mut rng = thread_rng();
        let (lo, hi) = self.factor_range;
        let factor = lo + rng.random::<f32>() * (hi - lo);
        let shape = input.shape().dims().to_vec();
        let device = input.device();
        let data = input
            .to_vec()
            .map_err(|e| torsh_core::error::TorshError::Other(format!("to_vec failed: {}", e)))?;
        let n = data.len();
        let mean: f64 = data
            .iter()
            .map(|x| <T as TensorElement>::to_f64(x).unwrap_or(0.0))
            .sum::<f64>()
            / n.max(1) as f64;
        let out: Vec<T> = data
            .iter()
            .map(|&x| {
                let v =
                    mean + (<T as TensorElement>::to_f64(&x).unwrap_or(0.0) - mean) * factor as f64;
                let clamped = v.max(0.0).min(1.0);
                <T as TensorElement>::from_f64(clamped).unwrap_or(x)
            })
            .collect();
        Tensor::from_data(out, shape, device).map_err(|e| e.into())
    }
}

/// Random saturation adjustment (for color images)
pub struct RandomSaturation {
    factor_range: (f32, f32),
}

impl RandomSaturation {
    pub fn new(factor_range: (f32, f32)) -> Self {
        assert!(factor_range.0 <= factor_range.1, "Invalid factor range");
        Self { factor_range }
    }

    /// Create with symmetric range around 1.0
    pub fn symmetric(factor: f32) -> Self {
        Self::new((1.0 - factor, 1.0 + factor))
    }
}

impl<T: FloatElement> Transform<Tensor<T>> for RandomSaturation {
    type Output = Tensor<T>;

    fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
        let mut rng = thread_rng();
        let (lo, hi) = self.factor_range;
        let factor = lo + rng.random::<f32>() * (hi - lo);
        let binding = input.shape();
        let dims = binding.dims();
        // Only apply saturation for 3-channel CHW images
        if dims.len() != 3 || dims[0] != 3 {
            return Ok(input);
        }
        let (_, height, width) = (dims[0], dims[1], dims[2]);
        let hw = height * width;
        let device = input.device();
        let data = input
            .to_vec()
            .map_err(|e| torsh_core::error::TorshError::Other(format!("to_vec failed: {}", e)))?;
        // ITU-R BT.601 luminance coefficients
        const LUM_R: f64 = 0.299;
        const LUM_G: f64 = 0.587;
        const LUM_B: f64 = 0.114;
        let mut out = data.clone();
        for px in 0..hw {
            let r = <T as TensorElement>::to_f64(&data[px]).unwrap_or(0.0);
            let g = <T as TensorElement>::to_f64(&data[hw + px]).unwrap_or(0.0);
            let b = <T as TensorElement>::to_f64(&data[2 * hw + px]).unwrap_or(0.0);
            let lum = LUM_R * r + LUM_G * g + LUM_B * b;
            let sat = factor as f64;
            let new_r = (lum + sat * (r - lum)).max(0.0).min(1.0);
            let new_g = (lum + sat * (g - lum)).max(0.0).min(1.0);
            let new_b = (lum + sat * (b - lum)).max(0.0).min(1.0);
            out[px] = <T as TensorElement>::from_f64(new_r).unwrap_or(data[px]);
            out[hw + px] = <T as TensorElement>::from_f64(new_g).unwrap_or(data[hw + px]);
            out[2 * hw + px] = <T as TensorElement>::from_f64(new_b).unwrap_or(data[2 * hw + px]);
        }
        Tensor::from_data(out, dims.to_vec(), device).map_err(|e| e.into())
    }
}

/// Random hue adjustment (for color images)
pub struct RandomHue {
    delta_range: (f32, f32),
}

impl RandomHue {
    pub fn new(delta_range: (f32, f32)) -> Self {
        assert!(delta_range.0 <= delta_range.1, "Invalid delta range");
        assert!(
            delta_range.0 >= -1.0 && delta_range.1 <= 1.0,
            "Hue delta must be in [-1, 1]"
        );
        Self { delta_range }
    }

    /// Create with symmetric range
    pub fn symmetric(delta: f32) -> Self {
        Self::new((-delta, delta))
    }
}

impl<T: FloatElement> Transform<Tensor<T>> for RandomHue {
    type Output = Tensor<T>;

    fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
        // Note: True hue rotation requires RGB→HSV→RGB conversion.
        // This is a per-channel scale approximation: each channel is
        // scaled by a slightly different random factor derived from the
        // hue delta, which shifts the apparent color balance.
        let mut rng = thread_rng();
        let (lo, hi) = self.delta_range;
        let delta = lo + rng.random::<f32>() * (hi - lo);
        let binding = input.shape();
        let dims = binding.dims();
        // Only apply to 3-channel CHW images
        if dims.len() != 3 || dims[0] != 3 {
            return Ok(input);
        }
        let (_, height, width) = (dims[0], dims[1], dims[2]);
        let hw = height * width;
        let device = input.device();
        let data = input
            .to_vec()
            .map_err(|e| torsh_core::error::TorshError::Other(format!("to_vec failed: {}", e)))?;
        // Hue shift: rotate color weights using delta as rotation angle proxy
        // R channel boosted by sin(delta*pi), G unchanged, B reduced by sin(delta*pi)
        let angle = delta as f64 * std::f64::consts::PI;
        let r_scale = 1.0 + angle.sin() * 0.5;
        let g_scale = 1.0 - angle.abs().sin() * 0.1;
        let b_scale = 1.0 - angle.sin() * 0.5;
        let mut out = data.clone();
        for px in 0..hw {
            let r = <T as TensorElement>::to_f64(&data[px]).unwrap_or(0.0);
            let g = <T as TensorElement>::to_f64(&data[hw + px]).unwrap_or(0.0);
            let b = <T as TensorElement>::to_f64(&data[2 * hw + px]).unwrap_or(0.0);
            out[px] =
                <T as TensorElement>::from_f64((r * r_scale).max(0.0).min(1.0)).unwrap_or(data[px]);
            out[hw + px] = <T as TensorElement>::from_f64((g * g_scale).max(0.0).min(1.0))
                .unwrap_or(data[hw + px]);
            out[2 * hw + px] = <T as TensorElement>::from_f64((b * b_scale).max(0.0).min(1.0))
                .unwrap_or(data[2 * hw + px]);
        }
        Tensor::from_data(out, dims.to_vec(), device).map_err(|e| e.into())
    }
}

/// Random vertical flip
pub struct RandomVerticalFlip {
    prob: f32,
}

impl RandomVerticalFlip {
    pub fn new(prob: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability must be between 0 and 1"
        );
        Self { prob }
    }
}

impl<T: FloatElement> Transform<Tensor<T>> for RandomVerticalFlip {
    type Output = Tensor<T>;

    fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
        let mut rng = thread_rng();
        if rng.random::<f32>() >= self.prob {
            return Ok(input);
        }
        let binding = input.shape();
        let dims = binding.dims();
        if dims.len() < 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Input tensor must have at least 2 dimensions for vertical flip".to_string(),
            ));
        }
        let device = input.device();
        let data = input
            .to_vec()
            .map_err(|e| torsh_core::error::TorshError::Other(format!("to_vec failed: {}", e)))?;
        // For CHW (3D+), flip dim[-2] (height); for HW (2D), flip dim 0 (height)
        let (height, width, channels) = if dims.len() == 2 {
            (dims[0], dims[1], 1usize)
        } else {
            (
                dims[dims.len() - 2],
                dims[dims.len() - 1],
                dims[..dims.len() - 2].iter().product(),
            )
        };
        let mut out = data.clone();
        for c in 0..channels {
            for row in 0..height / 2 {
                let mirror_row = height - 1 - row;
                for col in 0..width {
                    let idx1 = c * height * width + row * width + col;
                    let idx2 = c * height * width + mirror_row * width + col;
                    out.swap(idx1, idx2);
                }
            }
        }
        Tensor::from_data(out, dims.to_vec(), device).map_err(|e| e.into())
    }
}

/// Gaussian noise addition
pub struct GaussianNoise {
    mean: f32,
    std: f32,
}

impl GaussianNoise {
    pub fn new(mean: f32, std: f32) -> Self {
        assert!(std >= 0.0, "Standard deviation must be non-negative");
        Self { mean, std }
    }

    /// Create with zero mean
    pub fn with_std(std: f32) -> Self {
        Self::new(0.0, std)
    }
}

impl<T: FloatElement> Transform<Tensor<T>> for GaussianNoise {
    type Output = Tensor<T>;

    fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
        if self.std <= 0.0 {
            return Ok(input);
        }
        let mut rng = thread_rng();
        let shape = input.shape().dims().to_vec();
        let device = input.device();
        let data = input
            .to_vec()
            .map_err(|e| torsh_core::error::TorshError::Other(format!("to_vec failed: {}", e)))?;
        let mean_f64 = self.mean as f64;
        let std_f64 = self.std as f64;
        let out: Vec<T> = data
            .iter()
            .map(|&x| {
                // Box-Muller transform for Gaussian noise
                let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
                let u2: f32 = rng.random::<f32>();
                let noise = ((-2.0 * u1.ln()) as f64).sqrt()
                    * (2.0 * std::f64::consts::PI * u2 as f64).cos();
                let noisy =
                    <T as TensorElement>::to_f64(&x).unwrap_or(0.0) + mean_f64 + std_f64 * noise;
                <T as TensorElement>::from_f64(noisy).unwrap_or(x)
            })
            .collect();
        Tensor::from_data(out, shape, device).map_err(|e| e.into())
    }
}

/// Random erasing (cutout) augmentation
pub struct RandomErasing {
    prob: f32,
    scale_range: (f32, f32),
    ratio_range: (f32, f32),
    fill_value: f32,
}

impl RandomErasing {
    pub fn new(prob: f32, scale_range: (f32, f32), ratio_range: (f32, f32)) -> Self {
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability must be between 0 and 1"
        );
        assert!(scale_range.0 <= scale_range.1, "Invalid scale range");
        assert!(ratio_range.0 <= ratio_range.1, "Invalid ratio range");

        Self {
            prob,
            scale_range,
            ratio_range,
            fill_value: 0.0,
        }
    }

    pub fn with_fill_value(mut self, fill_value: f32) -> Self {
        self.fill_value = fill_value;
        self
    }
}

impl<T: FloatElement> Transform<Tensor<T>> for RandomErasing {
    type Output = Tensor<T>;

    fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
        let mut rng = thread_rng();
        if rng.random::<f32>() >= self.prob {
            return Ok(input);
        }
        let binding = input.shape();
        let dims = binding.dims();
        if dims.len() < 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Input tensor must have at least 2 dimensions for random erasing".to_string(),
            ));
        }
        let device = input.device();
        let (height, width, channels) = if dims.len() == 2 {
            (dims[0], dims[1], 1usize)
        } else {
            (
                dims[dims.len() - 2],
                dims[dims.len() - 1],
                dims[..dims.len() - 2].iter().product(),
            )
        };
        let total_area = (height * width) as f32;
        // Sample erasing area as fraction of total
        let (scale_lo, scale_hi) = self.scale_range;
        let area_frac = scale_lo + rng.random::<f32>() * (scale_hi - scale_lo);
        let erase_area = (total_area * area_frac) as usize;
        // Sample aspect ratio
        let (ratio_lo, ratio_hi) = self.ratio_range;
        let ratio = ratio_lo + rng.random::<f32>() * (ratio_hi - ratio_lo);
        let erase_h = ((erase_area as f32 / ratio).sqrt() as usize).clamp(1, height);
        let erase_w = ((erase_area as f32 * ratio).sqrt() as usize).clamp(1, width);
        if erase_h >= height || erase_w >= width {
            return Ok(input);
        }
        let top = rng.gen_range(0..=(height - erase_h));
        let left = rng.gen_range(0..=(width - erase_w));
        let fill = <T as TensorElement>::from_f64(self.fill_value as f64)
            .unwrap_or_else(<T as TensorElement>::zero);
        let mut data = input
            .to_vec()
            .map_err(|e| torsh_core::error::TorshError::Other(format!("to_vec failed: {}", e)))?;
        for c in 0..channels {
            for row in top..(top + erase_h) {
                for col in left..(left + erase_w) {
                    let idx = c * height * width + row * width + col;
                    data[idx] = fill;
                }
            }
        }
        Tensor::from_data(data, dims.to_vec(), device).map_err(|e| e.into())
    }
}

/// Common augmentation presets
impl AugmentationPipeline<Tensor<f32>> {
    /// Create a light augmentation pipeline for training
    pub fn light_augmentation() -> Self {
        Self::new()
            .add_conditional(
                crate::tensor_transforms::RandomHorizontalFlip::new(0.5),
                1.0,
            )
            .add_conditional(RandomBrightness::symmetric(0.1), 0.3)
            .add_conditional(RandomContrast::symmetric(0.1), 0.3)
    }

    /// Create a medium augmentation pipeline
    pub fn medium_augmentation() -> Self {
        Self::new()
            .add_conditional(
                crate::tensor_transforms::RandomHorizontalFlip::new(0.5),
                1.0,
            )
            .add_conditional(RandomVerticalFlip::new(0.1), 1.0)
            .add_conditional(RandomBrightness::symmetric(0.2), 0.5)
            .add_conditional(RandomContrast::symmetric(0.2), 0.5)
            .add_conditional(RandomSaturation::symmetric(0.2), 0.3)
            .add_conditional(GaussianNoise::with_std(0.01), 0.2)
    }

    /// Create a heavy augmentation pipeline
    pub fn heavy_augmentation() -> Self {
        Self::new()
            .add_conditional(
                crate::tensor_transforms::RandomHorizontalFlip::new(0.5),
                1.0,
            )
            .add_conditional(RandomVerticalFlip::new(0.2), 1.0)
            .add_conditional(RandomBrightness::symmetric(0.3), 0.7)
            .add_conditional(RandomContrast::symmetric(0.3), 0.7)
            .add_conditional(RandomSaturation::symmetric(0.3), 0.5)
            .add_conditional(RandomHue::symmetric(0.1), 0.3)
            .add_conditional(GaussianNoise::with_std(0.02), 0.3)
            .add_conditional(RandomErasing::new(0.5, (0.02, 0.33), (0.3, 3.3)), 1.0)
    }

    /// Create an augmentation pipeline for ImageNet-style training
    pub fn imagenet_augmentation() -> Self {
        Self::new()
            .add_conditional(
                crate::tensor_transforms::RandomHorizontalFlip::new(0.5),
                1.0,
            )
            .add_conditional(RandomBrightness::symmetric(0.2), 0.4)
            .add_conditional(RandomContrast::symmetric(0.2), 0.4)
            .add_conditional(RandomSaturation::symmetric(0.2), 0.4)
            .add_conditional(RandomHue::symmetric(0.1), 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::Tensor;

    // Mock tensor for testing
    fn mock_tensor() -> Tensor<f32> {
        Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap()
    }

    #[test]
    fn test_augmentation_pipeline_creation() {
        let pipeline = AugmentationPipeline::<i32>::new();
        assert_eq!(pipeline.probability, 1.0);
        assert_eq!(pipeline.transforms.len(), 0);
    }

    #[test]
    fn test_augmentation_pipeline_with_probability() {
        let pipeline = AugmentationPipeline::<i32>::new().with_probability(0.5);
        assert_eq!(pipeline.probability, 0.5);
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0 and 1")]
    fn test_invalid_probability() {
        AugmentationPipeline::<i32>::new().with_probability(1.5);
    }

    #[test]
    fn test_conditional_transform_creation() {
        let transform: ConditionalTransform<i32, _> =
            ConditionalTransform::new(crate::transforms::lambda(|x: i32| Ok(x * 2)), 0.5);
        assert_eq!(transform.probability, 0.5);
    }

    #[test]
    fn test_random_brightness_creation() {
        let brightness = RandomBrightness::new((0.8, 1.2));
        assert_eq!(brightness.factor_range, (0.8, 1.2));
    }

    #[test]
    fn test_random_brightness_symmetric() {
        let brightness = RandomBrightness::symmetric(0.2);
        assert_eq!(brightness.factor_range, (0.8, 1.2));
    }

    #[test]
    fn test_gaussian_noise_creation() {
        let noise = GaussianNoise::new(0.0, 0.1);
        assert_eq!(noise.mean, 0.0);
        assert_eq!(noise.std, 0.1);
    }

    #[test]
    fn test_gaussian_noise_with_std() {
        let noise = GaussianNoise::with_std(0.05);
        assert_eq!(noise.mean, 0.0);
        assert_eq!(noise.std, 0.05);
    }

    #[test]
    fn test_random_erasing_creation() {
        let erasing = RandomErasing::new(0.5, (0.02, 0.33), (0.3, 3.3));
        assert_eq!(erasing.prob, 0.5);
        assert_eq!(erasing.scale_range, (0.02, 0.33));
        assert_eq!(erasing.ratio_range, (0.3, 3.3));
        assert_eq!(erasing.fill_value, 0.0);
    }

    #[test]
    fn test_light_augmentation_preset() {
        let pipeline = AugmentationPipeline::light_augmentation();
        assert_eq!(pipeline.transforms.len(), 3);
    }

    #[test]
    fn test_medium_augmentation_preset() {
        let pipeline = AugmentationPipeline::medium_augmentation();
        assert_eq!(pipeline.transforms.len(), 6);
    }

    #[test]
    fn test_heavy_augmentation_preset() {
        let pipeline = AugmentationPipeline::heavy_augmentation();
        assert_eq!(pipeline.transforms.len(), 8);
    }

    #[test]
    fn test_augmentation_transform_shape_preserved() {
        let tensor = mock_tensor();
        let brightness = RandomBrightness::symmetric(0.1);
        let result = brightness.transform(tensor.clone()).unwrap();

        // Transforms operate on shape-preserving tensors
        assert_eq!(result.shape(), tensor.shape());
    }

    #[test]
    fn test_random_brightness_changes_tensor() {
        // factor range that never includes 1.0, so output always differs from input (all-1.0 tensor)
        let brightness = RandomBrightness::new((0.5, 0.7));
        let tensor = Tensor::from_data(vec![1.0f32; 4], vec![2, 2], DeviceType::Cpu).unwrap();
        let result = brightness.transform(tensor).unwrap();
        let result_data = result.to_vec().unwrap();
        // Should be between 0.5 and 0.7 (darkened from 1.0)
        assert!(
            result_data.iter().all(|&x| x < 1.0),
            "Brightness transform must darken the tensor (factor in [0.5, 0.7])"
        );
    }

    #[test]
    fn test_gaussian_noise_changes_tensor() {
        let noise = GaussianNoise::with_std(0.5);
        // Run multiple times to be nearly certain noise is non-zero
        let mut changed = false;
        for _ in 0..10 {
            let tensor = Tensor::from_data(vec![0.5f32; 16], vec![4, 4], DeviceType::Cpu).unwrap();
            let result = noise.transform(tensor).unwrap();
            let data = result.to_vec().unwrap();
            if data.iter().any(|&x| (x - 0.5f32).abs() > 1e-6) {
                changed = true;
                break;
            }
        }
        assert!(changed, "GaussianNoise must change tensor values");
    }

    #[test]
    fn test_random_vertical_flip_changes_tensor() {
        // Always flip (prob=1.0)
        let flip = RandomVerticalFlip::new(1.0);
        // HW tensor: [[1,2],[3,4]] -> after vertical flip -> [[3,4],[1,2]]
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let result = flip.transform(tensor).unwrap();
        let data = result.to_vec().unwrap();
        // After vertical flip: row 0 becomes [3,4], row 1 becomes [1,2]
        assert!(
            (data[0] - 3.0).abs() < 1e-6 && (data[1] - 4.0).abs() < 1e-6,
            "Vertical flip must reverse rows: got {:?}",
            data
        );
    }

    #[test]
    fn test_random_erasing_changes_tensor() {
        // Always erase (prob=1.0), large erase area
        let erasing = RandomErasing::new(1.0, (0.5, 0.9), (1.0, 1.0));
        // Large tensor of all 1.0s; after erasing, some should be 0.0 (fill_value)
        let tensor = Tensor::from_data(vec![1.0f32; 100], vec![10, 10], DeviceType::Cpu).unwrap();
        let result = erasing.transform(tensor).unwrap();
        let data = result.to_vec().unwrap();
        assert!(
            data.iter().any(|&x| x == 0.0f32),
            "RandomErasing must fill some values with fill_value (0.0)"
        );
    }
}
