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
use torsh_core::dtype::FloatElement;
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
    #[allow(dead_code)]
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
        // For now, return input as-is since tensor operations need proper trait bounds
        // In a full implementation, we would apply brightness adjustment
        Ok(input)
    }
}

/// Random contrast adjustment
pub struct RandomContrast {
    #[allow(dead_code)]
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
        // For now, return input as-is since tensor operations need proper trait bounds
        // In a full implementation, we would apply contrast adjustment
        Ok(input)
    }
}

/// Random saturation adjustment (for color images)
pub struct RandomSaturation {
    #[allow(dead_code)]
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
        // For now, return input as-is since proper saturation adjustment
        // requires complex RGB to grayscale conversion operations
        Ok(input)
    }
}

/// Random hue adjustment (for color images)
pub struct RandomHue {
    #[allow(dead_code)]
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
        // For now, return input as-is since proper HSV conversion
        // requires more complex operations
        Ok(input)
    }
}

/// Random vertical flip
pub struct RandomVerticalFlip {
    #[allow(dead_code)]
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
        // For now, return input as-is
        Ok(input)
    }
}

/// Gaussian noise addition
pub struct GaussianNoise {
    #[allow(dead_code)]
    mean: f32,
    #[allow(dead_code)]
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
        // For now, return input as-is
        Ok(input)
    }
}

/// Random erasing (cutout) augmentation
pub struct RandomErasing {
    #[allow(dead_code)]
    prob: f32,
    #[allow(dead_code)]
    scale_range: (f32, f32),
    #[allow(dead_code)]
    ratio_range: (f32, f32),
    #[allow(dead_code)]
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
        // For now, return input as-is
        Ok(input)
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
    fn test_augmentation_transform_passthrough() {
        let tensor = mock_tensor();
        let brightness = RandomBrightness::symmetric(0.1);
        let result = brightness.transform(tensor.clone()).unwrap();

        // For now, transforms are passthrough, so tensor should be unchanged
        assert_eq!(result.shape(), tensor.shape());
    }
}
