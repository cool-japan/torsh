use super::augmentation::{ColorJitter, RandomErasing};
use super::basic::{CenterCrop, Normalize, Pad, Resize};
use super::core::Compose;
use super::random::{RandomCrop, RandomHorizontalFlip};
use super::registry::TransformBuilder;

/// Convenient constructor functions for common transform combinations
pub mod presets {
    use super::*;

    /// ImageNet training transforms
    ///
    /// Standard ImageNet preprocessing pipeline for training:
    /// 1. Resize to larger size
    /// 2. Random horizontal flip
    /// 3. Center crop to target size
    /// 4. ImageNet normalization
    pub fn imagenet_train(size: usize) -> Compose {
        TransformBuilder::new()
            .resize((size + 32, size + 32))
            .random_horizontal_flip(0.5)
            .center_crop((size, size))
            .imagenet_normalize()
            .build()
    }

    /// ImageNet validation transforms
    ///
    /// Standard ImageNet preprocessing pipeline for validation/inference:
    /// 1. Resize to target size
    /// 2. Center crop to target size
    /// 3. ImageNet normalization
    pub fn imagenet_val(size: usize) -> Compose {
        TransformBuilder::new()
            .resize((size, size))
            .center_crop((size, size))
            .imagenet_normalize()
            .build()
    }

    /// CIFAR training transforms
    ///
    /// Standard CIFAR-10/100 preprocessing pipeline for training:
    /// 1. Random horizontal flip
    /// 2. Padding
    /// 3. Random crop back to original size
    /// 4. CIFAR normalization
    pub fn cifar_train() -> Compose {
        TransformBuilder::new()
            .random_horizontal_flip(0.5)
            .add(Pad::symmetric(4, 0.0))
            .add(RandomCrop::new((32, 32)))
            .normalize(vec![0.4914, 0.4822, 0.4465], vec![0.2023, 0.1994, 0.2010])
            .build()
    }

    /// CIFAR validation transforms
    ///
    /// Standard CIFAR-10/100 preprocessing pipeline for validation/inference:
    /// 1. CIFAR normalization only
    pub fn cifar_val() -> Compose {
        TransformBuilder::new()
            .normalize(vec![0.4914, 0.4822, 0.4465], vec![0.2023, 0.1994, 0.2010])
            .build()
    }

    /// Strong augmentation for training
    ///
    /// Heavy augmentation pipeline for robust training:
    /// 1. Resize to larger size
    /// 2. Padding
    /// 3. Random crop
    /// 4. Random horizontal flip
    /// 5. Color jitter (brightness, contrast, saturation, hue)
    /// 6. Random erasing
    /// 7. ImageNet normalization
    pub fn strong_augment(size: usize) -> Compose {
        TransformBuilder::new()
            .resize((size + 32, size + 32))
            .add(Pad::symmetric(4, 0.0))
            .add(RandomCrop::new((size, size)))
            .random_horizontal_flip(0.5)
            .add(
                ColorJitter::new()
                    .brightness(0.4)
                    .contrast(0.4)
                    .saturation(0.4)
                    .hue(0.1),
            )
            .add(RandomErasing::new(0.25))
            .imagenet_normalize()
            .build()
    }

    /// Light augmentation for training
    ///
    /// Minimal augmentation pipeline for sensitive training:
    /// 1. Resize
    /// 2. Random horizontal flip
    /// 3. Light color jitter
    /// 4. ImageNet normalization
    pub fn light_augment(size: usize) -> Compose {
        TransformBuilder::new()
            .resize((size, size))
            .random_horizontal_flip(0.5)
            .add(ColorJitter::new().brightness(0.1).contrast(0.1))
            .imagenet_normalize()
            .build()
    }

    /// No augmentation baseline
    ///
    /// Minimal preprocessing without augmentation:
    /// 1. Resize to target size
    /// 2. ImageNet normalization
    pub fn no_augment(size: usize) -> Compose {
        TransformBuilder::new()
            .resize((size, size))
            .imagenet_normalize()
            .build()
    }

    /// Custom training preset with configurable parameters
    ///
    /// Flexible training pipeline with customizable augmentation strength
    pub fn custom_train(
        size: usize,
        flip_prob: f32,
        color_jitter_strength: f32,
        erase_prob: f32,
    ) -> Compose {
        let mut builder = TransformBuilder::new()
            .resize((size + 16, size + 16))
            .add(RandomCrop::new((size, size)))
            .random_horizontal_flip(flip_prob);

        if color_jitter_strength > 0.0 {
            builder = builder.add(
                ColorJitter::new()
                    .brightness(color_jitter_strength * 0.4)
                    .contrast(color_jitter_strength * 0.4)
                    .saturation(color_jitter_strength * 0.4)
                    .hue(color_jitter_strength * 0.1),
            );
        }

        if erase_prob > 0.0 {
            builder = builder.add(RandomErasing::new(erase_prob));
        }

        builder.imagenet_normalize().build()
    }

    /// Medical imaging preset
    ///
    /// Conservative augmentation for medical images:
    /// 1. Resize
    /// 2. Light rotation and flip
    /// 3. Minimal color adjustment
    /// 4. Normalization with medical imaging statistics
    pub fn medical_imaging(size: usize) -> Compose {
        TransformBuilder::new()
            .resize((size, size))
            .random_horizontal_flip(0.3)
            .add(ColorJitter::new().brightness(0.05).contrast(0.05))
            .normalize(vec![0.5], vec![0.5]) // Grayscale normalization
            .build()
    }

    /// Object detection preset
    ///
    /// Augmentation suitable for object detection tasks:
    /// 1. Resize with aspect ratio preservation
    /// 2. Random horizontal flip
    /// 3. Light color jitter
    /// 4. ImageNet normalization
    pub fn object_detection(size: usize) -> Compose {
        TransformBuilder::new()
            .resize((size, size))
            .random_horizontal_flip(0.5)
            .add(
                ColorJitter::new()
                    .brightness(0.2)
                    .contrast(0.2)
                    .saturation(0.2),
            )
            .imagenet_normalize()
            .build()
    }

    /// Fine-tuning preset
    ///
    /// Conservative augmentation for fine-tuning pre-trained models:
    /// 1. Resize
    /// 2. Light augmentation
    /// 3. ImageNet normalization (maintains pre-training statistics)
    pub fn fine_tuning(size: usize) -> Compose {
        TransformBuilder::new()
            .resize((size, size))
            .random_horizontal_flip(0.3)
            .add(ColorJitter::new().brightness(0.1))
            .imagenet_normalize()
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::presets::*;

    #[test]
    fn test_imagenet_train() {
        let transforms = imagenet_train(224);
        assert_eq!(transforms.len(), 4);
        assert_eq!(transforms.transforms()[0].name(), "Resize");
        assert_eq!(transforms.transforms()[1].name(), "RandomHorizontalFlip");
        assert_eq!(transforms.transforms()[2].name(), "CenterCrop");
        assert_eq!(transforms.transforms()[3].name(), "Normalize");
    }

    #[test]
    fn test_imagenet_val() {
        let transforms = imagenet_val(224);
        assert_eq!(transforms.len(), 3);
        assert_eq!(transforms.transforms()[0].name(), "Resize");
        assert_eq!(transforms.transforms()[1].name(), "CenterCrop");
        assert_eq!(transforms.transforms()[2].name(), "Normalize");
    }

    #[test]
    fn test_cifar_train() {
        let transforms = cifar_train();
        assert_eq!(transforms.len(), 4);
        assert_eq!(transforms.transforms()[0].name(), "RandomHorizontalFlip");
        assert_eq!(transforms.transforms()[1].name(), "Pad");
        assert_eq!(transforms.transforms()[2].name(), "RandomCrop");
        assert_eq!(transforms.transforms()[3].name(), "Normalize");
    }

    #[test]
    fn test_cifar_val() {
        let transforms = cifar_val();
        assert_eq!(transforms.len(), 1);
        assert_eq!(transforms.transforms()[0].name(), "Normalize");
    }

    #[test]
    fn test_strong_augment() {
        let transforms = strong_augment(224);
        assert_eq!(transforms.len(), 7);

        // Check key transforms are present
        let names: Vec<&str> = transforms.transforms().iter().map(|t| t.name()).collect();
        assert!(names.contains(&"Resize"));
        assert!(names.contains(&"Pad"));
        assert!(names.contains(&"RandomCrop"));
        assert!(names.contains(&"RandomHorizontalFlip"));
        assert!(names.contains(&"ColorJitter"));
        assert!(names.contains(&"RandomErasing"));
        assert!(names.contains(&"Normalize"));
    }

    #[test]
    fn test_light_augment() {
        let transforms = light_augment(224);
        assert_eq!(transforms.len(), 4);
        assert_eq!(transforms.transforms()[0].name(), "Resize");
        assert_eq!(transforms.transforms()[1].name(), "RandomHorizontalFlip");
        assert_eq!(transforms.transforms()[2].name(), "ColorJitter");
        assert_eq!(transforms.transforms()[3].name(), "Normalize");
    }

    #[test]
    fn test_no_augment() {
        let transforms = no_augment(224);
        assert_eq!(transforms.len(), 2);
        assert_eq!(transforms.transforms()[0].name(), "Resize");
        assert_eq!(transforms.transforms()[1].name(), "Normalize");
    }

    #[test]
    fn test_custom_train_no_augment() {
        let transforms = custom_train(224, 0.0, 0.0, 0.0);
        assert_eq!(transforms.len(), 4); // resize, crop, flip, normalize

        let names: Vec<&str> = transforms.transforms().iter().map(|t| t.name()).collect();
        assert!(names.contains(&"Resize"));
        assert!(names.contains(&"RandomCrop"));
        assert!(names.contains(&"RandomHorizontalFlip"));
        assert!(names.contains(&"Normalize"));
    }

    #[test]
    fn test_custom_train_full_augment() {
        let transforms = custom_train(224, 0.5, 1.0, 0.2);
        assert_eq!(transforms.len(), 6); // resize, crop, flip, color jitter, erasing, normalize

        let names: Vec<&str> = transforms.transforms().iter().map(|t| t.name()).collect();
        assert!(names.contains(&"ColorJitter"));
        assert!(names.contains(&"RandomErasing"));
    }

    #[test]
    fn test_medical_imaging() {
        let transforms = medical_imaging(256);
        assert_eq!(transforms.len(), 4);
        assert_eq!(transforms.transforms()[0].name(), "Resize");
        assert_eq!(transforms.transforms()[1].name(), "RandomHorizontalFlip");
        assert_eq!(transforms.transforms()[2].name(), "ColorJitter");
        assert_eq!(transforms.transforms()[3].name(), "Normalize");
    }

    #[test]
    fn test_object_detection() {
        let transforms = object_detection(416);
        assert_eq!(transforms.len(), 4);
        assert_eq!(transforms.transforms()[0].name(), "Resize");
        assert_eq!(transforms.transforms()[1].name(), "RandomHorizontalFlip");
        assert_eq!(transforms.transforms()[2].name(), "ColorJitter");
        assert_eq!(transforms.transforms()[3].name(), "Normalize");
    }

    #[test]
    fn test_fine_tuning() {
        let transforms = fine_tuning(224);
        assert_eq!(transforms.len(), 4);
        assert_eq!(transforms.transforms()[0].name(), "Resize");
        assert_eq!(transforms.transforms()[1].name(), "RandomHorizontalFlip");
        assert_eq!(transforms.transforms()[2].name(), "ColorJitter");
        assert_eq!(transforms.transforms()[3].name(), "Normalize");
    }

    #[test]
    fn test_all_presets_different_sizes() {
        // Test that presets work with different input sizes
        let sizes = [128, 224, 256, 384];

        for &size in &sizes {
            let train = imagenet_train(size);
            let val = imagenet_val(size);
            let strong = strong_augment(size);
            let light = light_augment(size);
            let no_aug = no_augment(size);

            // All should be valid (non-empty)
            assert!(!train.is_empty());
            assert!(!val.is_empty());
            assert!(!strong.is_empty());
            assert!(!light.is_empty());
            assert!(!no_aug.is_empty());
        }
    }
}
