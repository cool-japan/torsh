//! Self-Supervised Learning Augmentation Strategies
//!
//! This module provides augmentation pipelines specifically designed for
//! self-supervised learning methods like SimCLR, MoCo, BYOL, SwAV, etc.
//!
//! These methods learn visual representations without labels by creating
//! different augmented views of the same image and training the model to
//! recognize them as similar.

use crate::{
    ColorJitter, Compose, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, Result, Transform,
};
use scirs2_core::random::{thread_rng, Random}; // SciRS2 Policy compliance
use torsh_tensor::Tensor;

// Re-export for convenience
pub(crate) type TensorF32 = Tensor<f32>;

/// SimCLR augmentation pipeline
///
/// SimCLR (Simple Framework for Contrastive Learning of Visual Representations)
/// uses strong data augmentation to create two correlated views of the same image.
///
/// Reference: Chen et al., "A Simple Framework for Contrastive Learning of
/// Visual Representations", ICML 2020
///
/// Key augmentations:
/// 1. Random cropping followed by resize back to original size
/// 2. Random color distortions
/// 3. Random Gaussian blur
/// 4. Random horizontal flip
pub struct SimCLRAugmentation {
    crop_size: usize,
    color_strength: f32,
    blur_probability: f32,
    transform: Box<dyn Transform>,
}

impl SimCLRAugmentation {
    /// Create SimCLR augmentation pipeline
    ///
    /// # Arguments
    /// * `crop_size` - Target size for cropped images
    /// * `color_strength` - Strength of color jitter (0.0 to 1.0)
    /// * `blur_probability` - Probability of applying Gaussian blur
    pub fn new(crop_size: usize, color_strength: f32, blur_probability: f32) -> Self {
        let transform = Box::new(Compose::new(vec![
            // Random cropping with resize
            Box::new(RandomResizedCrop::new((crop_size, crop_size))),
            // Random horizontal flip
            Box::new(RandomHorizontalFlip::new(0.5)),
            // Strong color jitter
            Box::new(
                ColorJitter::new()
                    .brightness(0.8 * color_strength)
                    .contrast(0.8 * color_strength)
                    .saturation(0.8 * color_strength)
                    .hue(0.2 * color_strength),
            ),
            // Random grayscale with 20% probability
            Box::new(RandomGrayscale::new(0.2)),
            // Gaussian blur (applied with specified probability)
            Box::new(GaussianBlur::new(
                (crop_size / 10).max(3) | 1,
                blur_probability,
            )),
        ]));

        Self {
            crop_size,
            color_strength,
            blur_probability,
            transform,
        }
    }

    /// Generate two augmented views of the same image
    ///
    /// # Arguments
    /// * `image` - Input image tensor
    ///
    /// # Returns
    /// Tuple of (view1, view2) - two differently augmented versions
    pub fn generate_pair(&self, image: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let view1 = self.transform.forward(image)?;
        let view2 = self.transform.forward(image)?;
        Ok((view1, view2))
    }

    /// Generate multiple augmented views
    pub fn generate_views(
        &self,
        image: &Tensor<f32>,
        num_views: usize,
    ) -> Result<Vec<Tensor<f32>>> {
        let mut views = Vec::with_capacity(num_views);
        for _ in 0..num_views {
            views.push(self.transform.forward(image)?);
        }
        Ok(views)
    }
}

/// MoCo (Momentum Contrast) augmentation
///
/// MoCo uses similar augmentations to SimCLR but typically with slightly
/// weaker augmentations for the query and stronger for the key.
///
/// Reference: He et al., "Momentum Contrast for Unsupervised Visual
/// Representation Learning", CVPR 2020
pub struct MoCoAugmentation {
    query_transform: Box<dyn Transform>,
    key_transform: Box<dyn Transform>,
}

impl MoCoAugmentation {
    /// Create MoCo augmentation with separate query and key transforms
    pub fn new(crop_size: usize) -> Self {
        // Query: weaker augmentation
        let query_transform = Box::new(Compose::new(vec![
            Box::new(RandomResizedCrop::new((crop_size, crop_size))),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(
                ColorJitter::new()
                    .brightness(0.4)
                    .contrast(0.4)
                    .saturation(0.4)
                    .hue(0.1),
            ),
        ]));

        // Key: similar augmentation (in MoCo v2)
        let key_transform = Box::new(Compose::new(vec![
            Box::new(RandomResizedCrop::new((crop_size, crop_size))),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(
                ColorJitter::new()
                    .brightness(0.4)
                    .contrast(0.4)
                    .saturation(0.4)
                    .hue(0.1),
            ),
            Box::new(RandomGrayscale::new(0.2)),
        ]));

        Self {
            query_transform,
            key_transform,
        }
    }

    /// Generate query and key views
    pub fn generate_pair(&self, image: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let query = self.query_transform.forward(image)?;
        let key = self.key_transform.forward(image)?;
        Ok((query, key))
    }
}

/// BYOL (Bootstrap Your Own Latent) augmentation
///
/// BYOL uses asymmetric augmentation strategies where one view is augmented
/// more strongly than the other.
///
/// Reference: Grill et al., "Bootstrap Your Own Latent: A New Approach to
/// Self-Supervised Learning", NeurIPS 2020
pub struct BYOLAugmentation {
    online_transform: Box<dyn Transform>,
    target_transform: Box<dyn Transform>,
}

impl BYOLAugmentation {
    /// Create BYOL augmentation with asymmetric transforms
    pub fn new(crop_size: usize) -> Self {
        // Online network: stronger augmentation
        let online_transform = Box::new(Compose::new(vec![
            Box::new(RandomResizedCrop::new((crop_size, crop_size))),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(
                ColorJitter::new()
                    .brightness(0.4)
                    .contrast(0.4)
                    .saturation(0.2)
                    .hue(0.1),
            ),
            Box::new(RandomGrayscale::new(0.2)),
            Box::new(GaussianBlur::new((crop_size / 10).max(3) | 1, 1.0)), // Always blur
            Box::new(Solarize::new(0.0, 0.0)), // Simplified solarization
        ]));

        // Target network: different augmentation
        let target_transform = Box::new(Compose::new(vec![
            Box::new(RandomResizedCrop::new((crop_size, crop_size))),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(
                ColorJitter::new()
                    .brightness(0.4)
                    .contrast(0.4)
                    .saturation(0.2)
                    .hue(0.1),
            ),
            Box::new(RandomGrayscale::new(0.2)),
            Box::new(GaussianBlur::new((crop_size / 10).max(3) | 1, 0.1)), // Rarely blur
        ]));

        Self {
            online_transform,
            target_transform,
        }
    }

    /// Generate online and target views
    pub fn generate_pair(&self, image: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let online_view = self.online_transform.forward(image)?;
        let target_view = self.target_transform.forward(image)?;
        Ok((online_view, target_view))
    }
}

/// SwAV (Swapping Assignments between Views) augmentation
///
/// SwAV uses multi-crop augmentation strategy with both global and local views.
///
/// Reference: Caron et al., "Unsupervised Learning of Visual Features by
/// Contrasting Cluster Assignments", NeurIPS 2020
#[derive(Debug)]
pub struct SwAVAugmentation {
    global_crop_size: usize,
    local_crop_size: usize,
    num_global_crops: usize,
    num_local_crops: usize,
}

impl SwAVAugmentation {
    /// Create SwAV multi-crop augmentation
    ///
    /// # Arguments
    /// * `global_crop_size` - Size for global crops (e.g., 224)
    /// * `local_crop_size` - Size for local crops (e.g., 96)
    /// * `num_global_crops` - Number of global views (typically 2)
    /// * `num_local_crops` - Number of local views (typically 6)
    pub fn new(
        global_crop_size: usize,
        local_crop_size: usize,
        num_global_crops: usize,
        num_local_crops: usize,
    ) -> Self {
        Self {
            global_crop_size,
            local_crop_size,
            num_global_crops,
            num_local_crops,
        }
    }

    /// Generate multi-crop views (global + local)
    pub fn generate_views(&self, image: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> {
        let mut views = Vec::new();

        // Global crops (cover >50% of image)
        let global_transform = self.create_global_transform();
        for _ in 0..self.num_global_crops {
            views.push(global_transform.forward(image)?);
        }

        // Local crops (cover <50% of image)
        let local_transform = self.create_local_transform();
        for _ in 0..self.num_local_crops {
            views.push(local_transform.forward(image)?);
        }

        Ok(views)
    }

    fn create_global_transform(&self) -> Box<dyn Transform> {
        Box::new(Compose::new(vec![
            Box::new(RandomResizedCrop::new((
                self.global_crop_size,
                self.global_crop_size,
            ))),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(
                ColorJitter::new()
                    .brightness(0.4)
                    .contrast(0.4)
                    .saturation(0.2)
                    .hue(0.1),
            ),
            Box::new(RandomGrayscale::new(0.2)),
            Box::new(GaussianBlur::new(
                (self.global_crop_size / 10).max(3) | 1,
                0.5,
            )),
        ]))
    }

    fn create_local_transform(&self) -> Box<dyn Transform> {
        Box::new(Compose::new(vec![
            Box::new(RandomResizedCrop::new((
                self.local_crop_size,
                self.local_crop_size,
            ))),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(
                ColorJitter::new()
                    .brightness(0.4)
                    .contrast(0.4)
                    .saturation(0.2)
                    .hue(0.1),
            ),
            Box::new(RandomGrayscale::new(0.2)),
            Box::new(GaussianBlur::new(
                (self.local_crop_size / 10).max(3) | 1,
                0.5,
            )),
        ]))
    }
}

/// DINO (Self-Distillation with No Labels) augmentation
///
/// DINO combines multi-crop strategy with strong augmentations for student
/// and weaker augmentations for teacher.
///
/// Reference: Caron et al., "Emerging Properties in Self-Supervised Vision
/// Transformers", ICCV 2021
#[derive(Debug)]
pub struct DINOAugmentation {
    global_crop_size: usize,
    local_crop_size: usize,
}

impl DINOAugmentation {
    /// Create DINO augmentation strategy
    pub fn new(global_crop_size: usize, local_crop_size: usize) -> Self {
        Self {
            global_crop_size,
            local_crop_size,
        }
    }

    /// Generate views for DINO (2 global + multiple local crops)
    pub fn generate_views(
        &self,
        image: &Tensor<f32>,
        num_local_crops: usize,
    ) -> Result<Vec<Tensor<f32>>> {
        let mut views = Vec::new();

        // 2 global crops (teacher and student)
        let global_transform = self.create_global_transform();
        views.push(global_transform.forward(image)?);
        views.push(global_transform.forward(image)?);

        // Multiple local crops (only for student)
        let local_transform = self.create_local_transform();
        for _ in 0..num_local_crops {
            views.push(local_transform.forward(image)?);
        }

        Ok(views)
    }

    fn create_global_transform(&self) -> Box<dyn Transform> {
        Box::new(Compose::new(vec![
            Box::new(RandomResizedCrop::new((
                self.global_crop_size,
                self.global_crop_size,
            ))),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(
                ColorJitter::new()
                    .brightness(0.4)
                    .contrast(0.4)
                    .saturation(0.2)
                    .hue(0.1),
            ),
            Box::new(RandomGrayscale::new(0.2)),
            Box::new(GaussianBlur::new(
                (self.global_crop_size / 10).max(3) | 1,
                1.0,
            )),
            Box::new(Solarize::new(0.0, 0.2)), // Solarize with 20% probability
        ]))
    }

    fn create_local_transform(&self) -> Box<dyn Transform> {
        Box::new(Compose::new(vec![
            Box::new(RandomResizedCrop::new((
                self.local_crop_size,
                self.local_crop_size,
            ))),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(
                ColorJitter::new()
                    .brightness(0.4)
                    .contrast(0.4)
                    .saturation(0.2)
                    .hue(0.1),
            ),
            Box::new(RandomGrayscale::new(0.2)),
            Box::new(GaussianBlur::new(
                (self.local_crop_size / 10).max(3) | 1,
                0.5,
            )),
        ]))
    }
}

/// Helper transforms for self-supervised learning
impl RandomGrayscale {
    /// Apply random grayscale conversion
    pub fn new(probability: f32) -> Self {
        Self { probability }
    }
}

/// Random grayscale transform
#[derive(Debug)]
pub struct RandomGrayscale {
    probability: f32,
}

impl Transform for RandomGrayscale {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut rng = thread_rng();
        if rng.random::<f32>() < self.probability {
            // Convert to grayscale and replicate to 3 channels
            let gray = crate::rgb_to_grayscale(input)?;
            // Replicate across channels by concatenating
            Ok(Tensor::cat(&[&gray, &gray, &gray], 0)?)
        } else {
            Ok(input.clone())
        }
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Self {
            probability: self.probability,
        })
    }
}

/// Gaussian blur transform
#[derive(Debug)]
pub struct GaussianBlur {
    kernel_size: usize,
    probability: f32,
}

impl GaussianBlur {
    pub fn new(kernel_size: usize, probability: f32) -> Self {
        Self {
            kernel_size,
            probability,
        }
    }
}

impl Transform for GaussianBlur {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut rng = thread_rng();
        if rng.random::<f32>() < self.probability {
            // Use appropriate sigma value
            let sigma = (self.kernel_size as f32) * 0.3;
            crate::gaussian_blur(input, sigma)
        } else {
            Ok(input.clone())
        }
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Self {
            kernel_size: self.kernel_size,
            probability: self.probability,
        })
    }
}

/// Solarization transform
#[derive(Debug)]
pub struct Solarize {
    threshold: f32,
    probability: f32,
}

impl Solarize {
    pub fn new(threshold: f32, probability: f32) -> Self {
        Self {
            threshold,
            probability,
        }
    }
}

impl Transform for Solarize {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut rng = thread_rng();
        if rng.random::<f32>() < self.probability {
            // Simplified solarization: invert pixel values
            let inverted = input.mul_scalar(-1.0)?.add_scalar(1.0)?;
            Ok(inverted)
        } else {
            Ok(input.clone())
        }
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Self {
            threshold: self.threshold,
            probability: self.probability,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::{self, randn};

    #[test]
    fn test_simclr_augmentation() {
        let aug = SimCLRAugmentation::new(224, 1.0, 0.5);
        assert_eq!(aug.crop_size, 224);
        assert_eq!(aug.color_strength, 1.0);
        assert_eq!(aug.blur_probability, 0.5);
    }

    #[test]
    fn test_simclr_generate_pair() {
        let aug = SimCLRAugmentation::new(224, 1.0, 0.5);
        let image = randn::<f32>(&[3, 256, 256]).unwrap();

        let result = aug.generate_pair(&image);
        assert!(result.is_ok());

        let (view1, view2) = result.unwrap();
        // Verify we got valid augmented views
        assert!(view1.numel() > 0);
        assert!(view2.numel() > 0);
    }

    #[test]
    fn test_moco_augmentation() {
        let aug = MoCoAugmentation::new(224);
        let image = randn::<f32>(&[3, 256, 256]).unwrap();

        let result = aug.generate_pair(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_byol_augmentation() {
        let aug = BYOLAugmentation::new(224);
        let image = randn::<f32>(&[3, 256, 256]).unwrap();

        let result = aug.generate_pair(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_swav_augmentation() {
        let aug = SwAVAugmentation::new(224, 96, 2, 6);
        let image = randn::<f32>(&[3, 256, 256]).unwrap();

        let result = aug.generate_views(&image);
        assert!(result.is_ok());

        let views = result.unwrap();
        assert_eq!(views.len(), 8); // 2 global + 6 local
    }

    #[test]
    fn test_dino_augmentation() {
        let aug = DINOAugmentation::new(224, 96);
        let image = randn::<f32>(&[3, 256, 256]).unwrap();

        let result = aug.generate_views(&image, 4);
        assert!(result.is_ok());

        let views = result.unwrap();
        assert_eq!(views.len(), 6); // 2 global + 4 local
    }

    #[test]
    fn test_random_grayscale() {
        let transform = RandomGrayscale::new(1.0); // Always apply
        let image = randn::<f32>(&[3, 64, 64]).unwrap();

        let result = transform.forward(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gaussian_blur() {
        let transform = GaussianBlur::new(5, 1.0);
        let image = randn::<f32>(&[3, 64, 64]).unwrap();

        let result = transform.forward(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_solarize() {
        let transform = Solarize::new(0.5, 1.0);
        let image = randn::<f32>(&[3, 64, 64]).unwrap();

        let result = transform.forward(&image);
        assert!(result.is_ok());
    }
}
