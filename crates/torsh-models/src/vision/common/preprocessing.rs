//! Image preprocessing utilities for vision models

use super::types::{ImageNormalization, VisionArchitecture};

/// Image preprocessor configuration
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    /// Target size for resizing
    pub target_size: (usize, usize),
    /// Normalization mean values (RGB)
    pub mean: [f32; 3],
    /// Normalization std values (RGB)
    pub std: [f32; 3],
    /// Whether to center crop
    pub center_crop: bool,
    /// Crop size if center cropping
    pub crop_size: Option<(usize, usize)>,
}

impl ImagePreprocessor {
    /// Create ImageNet preprocessor
    pub fn imagenet() -> Self {
        Self {
            target_size: (256, 256),
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            center_crop: true,
            crop_size: Some((224, 224)),
        }
    }

    /// Create CIFAR-10 preprocessor
    pub fn cifar10() -> Self {
        Self {
            target_size: (32, 32),
            mean: [0.4914, 0.4822, 0.4465],
            std: [0.2023, 0.1994, 0.2010],
            center_crop: false,
            crop_size: None,
        }
    }

    /// Create CIFAR-100 preprocessor
    pub fn cifar100() -> Self {
        Self {
            target_size: (32, 32),
            mean: [0.5071, 0.4867, 0.4408],
            std: [0.2675, 0.2565, 0.2761],
            center_crop: false,
            crop_size: None,
        }
    }

    /// Create custom preprocessor
    pub fn custom(
        target_size: (usize, usize),
        normalization: ImageNormalization,
        center_crop: bool,
        crop_size: Option<(usize, usize)>,
    ) -> Self {
        Self {
            target_size,
            mean: normalization.mean,
            std: normalization.std,
            center_crop,
            crop_size,
        }
    }

    /// Create preprocessor for specific architecture
    pub fn for_architecture(architecture: VisionArchitecture) -> Self {
        match architecture {
            VisionArchitecture::DETR | VisionArchitecture::MaskRCNN => Self::large_image(),
            VisionArchitecture::YOLO => Self::yolo(),
            _ => Self::imagenet(),
        }
    }

    /// Preprocessor for large image models (DETR, Mask R-CNN)
    pub fn large_image() -> Self {
        Self {
            target_size: (800, 800),
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            center_crop: false,
            crop_size: None,
        }
    }

    /// Preprocessor for YOLO models
    pub fn yolo() -> Self {
        Self {
            target_size: (640, 640),
            mean: [0.0, 0.0, 0.0], // YOLO typically uses [0, 1] normalization
            std: [255.0, 255.0, 255.0],
            center_crop: false,
            crop_size: None,
        }
    }

    /// Get the final output size after preprocessing
    pub fn output_size(&self) -> (usize, usize) {
        if let Some(crop_size) = self.crop_size {
            crop_size
        } else {
            self.target_size
        }
    }

    /// Get the number of preprocessing steps
    pub fn num_steps(&self) -> usize {
        let mut steps = 2; // resize + normalize
        if self.center_crop {
            steps += 1;
        }
        steps
    }
}

/// Data augmentation configuration
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    /// Random horizontal flip probability
    pub horizontal_flip: f32,
    /// Random vertical flip probability
    pub vertical_flip: f32,
    /// Random rotation angle range (degrees)
    pub rotation: Option<(f32, f32)>,
    /// Random scale range
    pub scale: Option<(f32, f32)>,
    /// Random crop size
    pub random_crop: Option<(usize, usize)>,
    /// Color jitter parameters (brightness, contrast, saturation, hue)
    pub color_jitter: Option<(f32, f32, f32, f32)>,
    /// Random erasing probability
    pub random_erasing: f32,
    /// Mixup alpha parameter
    pub mixup_alpha: Option<f32>,
    /// CutMix alpha parameter
    pub cutmix_alpha: Option<f32>,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            horizontal_flip: 0.5,
            vertical_flip: 0.0,
            rotation: None,
            scale: None,
            random_crop: None,
            color_jitter: None,
            random_erasing: 0.0,
            mixup_alpha: None,
            cutmix_alpha: None,
        }
    }
}

impl AugmentationConfig {
    /// Light augmentation for fine-tuning
    pub fn light() -> Self {
        Self {
            horizontal_flip: 0.5,
            color_jitter: Some((0.1, 0.1, 0.1, 0.05)),
            ..Default::default()
        }
    }

    /// Medium augmentation for training
    pub fn medium() -> Self {
        Self {
            horizontal_flip: 0.5,
            rotation: Some((-15.0, 15.0)),
            scale: Some((0.8, 1.2)),
            color_jitter: Some((0.2, 0.2, 0.2, 0.1)),
            random_erasing: 0.1,
            ..Default::default()
        }
    }

    /// Heavy augmentation for training from scratch
    pub fn heavy() -> Self {
        Self {
            horizontal_flip: 0.5,
            vertical_flip: 0.1,
            rotation: Some((-30.0, 30.0)),
            scale: Some((0.6, 1.4)),
            color_jitter: Some((0.3, 0.3, 0.3, 0.1)),
            random_erasing: 0.2,
            mixup_alpha: Some(0.2),
            cutmix_alpha: Some(1.0),
            ..Default::default()
        }
    }

    /// No augmentation
    pub fn none() -> Self {
        Self {
            horizontal_flip: 0.0,
            vertical_flip: 0.0,
            rotation: None,
            scale: None,
            random_crop: None,
            color_jitter: None,
            random_erasing: 0.0,
            mixup_alpha: None,
            cutmix_alpha: None,
        }
    }
}

/// Image preprocessing pipeline
#[derive(Debug, Clone)]
pub struct PreprocessingPipeline {
    /// Base preprocessor
    pub preprocessor: ImagePreprocessor,
    /// Data augmentation config (for training)
    pub augmentation: AugmentationConfig,
    /// Whether to apply augmentation
    pub training: bool,
}

impl PreprocessingPipeline {
    /// Create pipeline for training
    pub fn training(preprocessor: ImagePreprocessor, augmentation: AugmentationConfig) -> Self {
        Self {
            preprocessor,
            augmentation,
            training: true,
        }
    }

    /// Create pipeline for inference
    pub fn inference(preprocessor: ImagePreprocessor) -> Self {
        Self {
            preprocessor,
            augmentation: AugmentationConfig::none(),
            training: false,
        }
    }

    /// Get the final output size
    pub fn output_size(&self) -> (usize, usize) {
        self.preprocessor.output_size()
    }

    /// Switch to training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Switch to evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imagenet_preprocessor() {
        let preprocessor = ImagePreprocessor::imagenet();
        assert_eq!(preprocessor.target_size, (256, 256));
        assert_eq!(preprocessor.mean, [0.485, 0.456, 0.406]);
        assert_eq!(preprocessor.crop_size, Some((224, 224)));
        assert_eq!(preprocessor.output_size(), (224, 224));
    }

    #[test]
    fn test_cifar10_preprocessor() {
        let preprocessor = ImagePreprocessor::cifar10();
        assert_eq!(preprocessor.target_size, (32, 32));
        assert_eq!(preprocessor.mean, [0.4914, 0.4822, 0.4465]);
        assert!(!preprocessor.center_crop);
        assert_eq!(preprocessor.output_size(), (32, 32));
    }

    #[test]
    fn test_architecture_specific_preprocessor() {
        let detr_preprocessor = ImagePreprocessor::for_architecture(VisionArchitecture::DETR);
        assert_eq!(detr_preprocessor.output_size(), (800, 800));

        let yolo_preprocessor = ImagePreprocessor::for_architecture(VisionArchitecture::YOLO);
        assert_eq!(yolo_preprocessor.output_size(), (640, 640));

        let resnet_preprocessor = ImagePreprocessor::for_architecture(VisionArchitecture::ResNet);
        assert_eq!(resnet_preprocessor.output_size(), (224, 224));
    }

    #[test]
    fn test_augmentation_configs() {
        let light = AugmentationConfig::light();
        assert_eq!(light.horizontal_flip, 0.5);
        assert!(light.color_jitter.is_some());
        assert!(light.rotation.is_none());

        let heavy = AugmentationConfig::heavy();
        assert!(heavy.mixup_alpha.is_some());
        assert!(heavy.cutmix_alpha.is_some());
        assert!(heavy.rotation.is_some());
    }

    #[test]
    fn test_preprocessing_pipeline() {
        let preprocessor = ImagePreprocessor::imagenet();
        let augmentation = AugmentationConfig::medium();
        let mut pipeline = PreprocessingPipeline::training(preprocessor, augmentation);

        assert!(pipeline.training);
        pipeline.eval();
        assert!(!pipeline.training);

        assert_eq!(pipeline.output_size(), (224, 224));
    }
}
