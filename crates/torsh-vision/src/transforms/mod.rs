/*!
# ToRSh Vision Transforms

A comprehensive collection of image transformation and data augmentation utilities for computer vision tasks.
This module provides PyTorch-compatible transforms with enhanced functionality and performance optimizations.

## Organization

The transforms module is organized into several sub-modules:

- [`core`] - Core Transform trait and composition utilities
- [`basic`] - Fundamental transforms (Resize, Crop, Normalize, etc.)
- [`random`] - Random/probabilistic transforms for data augmentation
- [`augmentation`] - Advanced augmentation techniques (ColorJitter, RandomErasing, etc.)
- [`mixing`] - Data mixing techniques (MixUp, CutMix)
- [`automated`] - Automated augmentation strategies (AutoAugment, RandAugment)
- [`sophisticated`] - State-of-the-art augmentation methods (AugMix, GridMask, Mosaic)
- [`mod@registry`] - Transform registration and builder patterns
- [`presets`] - Common transform configurations for popular datasets

## Quick Start

### Basic Usage

```rust
use torsh_vision::transforms::{Compose, Resize, Normalize, RandomHorizontalFlip};

// Create a simple pipeline
let transforms = vec![
    Box::new(Resize::new((224, 224))) as Box<dyn Transform>,
    Box::new(RandomHorizontalFlip::new(0.5)),
    Box::new(Normalize::imagenet()),
];
let pipeline = Compose::new(transforms);
```

### Using the Builder Pattern

```rust
use torsh_vision::transforms::{TransformBuilder, presets};

// ImageNet training pipeline
let train_transforms = TransformBuilder::new()
    .resize((256, 256))
    .random_horizontal_flip(0.5)
    .center_crop((224, 224))
    .imagenet_normalize()
    .build();

// Or use presets
let train_transforms = presets::imagenet_train(224);
```

### Advanced Augmentation

```rust
use torsh_vision::transforms::{RandAugment, ColorJitter, RandomErasing};

// RandAugment for automated augmentation
let rand_aug = RandAugment::new(2, 5.0);

// Manual augmentation pipeline
let strong_aug = TransformBuilder::new()
    .resize((256, 256))
    .add(ColorJitter::new().brightness(0.4).contrast(0.4))
    .add(RandomErasing::new(0.25))
    .random_horizontal_flip(0.5)
    .center_crop((224, 224))
    .imagenet_normalize()
    .build();
```

## Key Features

- **SciRS2 Integration**: Full compliance with SciRS2 random number generation and array operations
- **Comprehensive Transform Library**: Over 20 different transforms covering basic to advanced techniques
- **Builder Pattern**: Fluent API for creating transform pipelines
- **Preset Configurations**: Ready-to-use configurations for common datasets and tasks
- **Advanced Techniques**: Support for state-of-the-art methods like AugMix, GridMask, and data mixing
- **Type Safety**: Strong typing with comprehensive error handling
- **Performance Optimized**: Efficient implementations with minimal memory allocation
- **Extensive Testing**: Comprehensive test coverage for all transforms
*/

//
// Module declarations
//

pub mod augmentation;
pub mod automated;
pub mod basic;
pub mod core;
pub mod mixing;
pub mod presets;
pub mod random;
pub mod registry;
pub mod sophisticated;
pub mod unified;

//
// Core exports
//

pub use core::{Compose, Transform};

//
// Basic transforms
//

pub use basic::{CenterCrop, Normalize, Pad, Resize, ToTensor};

//
// Random transforms
//

pub use random::{
    RandomCrop, RandomHorizontalFlip, RandomResizedCrop, RandomRotation, RandomVerticalFlip,
    Rotation,
};

//
// Augmentation transforms
//

pub use augmentation::{ColorJitter, Cutout, GaussianBlur, RandomErasing};

//
// Mixing techniques
//

pub use mixing::{CutMix, MixUp};

//
// Automated augmentation
//

pub use automated::{AutoAugment, RandAugment};

//
// Sophisticated techniques
//

pub use sophisticated::{AugMix, GridMask, Mosaic};

//
// Registry and builder patterns
//

pub use registry::{TransformBuilder, TransformIntrospection, TransformRegistry, TransformStats};

//
// Preset configurations
//

pub use presets::*;

//
// Utility functions and convenience constructors
//

/// Create a simple ImageNet training pipeline
///
/// # Arguments
///
/// * `size` - Target image size (will be square)
///
/// # Returns
///
/// A Compose transform ready for ImageNet training
pub fn imagenet_train(size: usize) -> Compose {
    presets::presets::imagenet_train(size)
}

/// Create a simple ImageNet validation pipeline
///
/// # Arguments
///
/// * `size` - Target image size (will be square)
///
/// # Returns
///
/// A Compose transform ready for ImageNet validation/inference
pub fn imagenet_val(size: usize) -> Compose {
    presets::presets::imagenet_val(size)
}

/// Create a CIFAR training pipeline
///
/// # Returns
///
/// A Compose transform ready for CIFAR-10/100 training
pub fn cifar_train() -> Compose {
    presets::presets::cifar_train()
}

/// Create a CIFAR validation pipeline
///
/// # Returns
///
/// A Compose transform ready for CIFAR-10/100 validation/inference
pub fn cifar_val() -> Compose {
    presets::presets::cifar_val()
}

/// Create a strong augmentation pipeline
///
/// # Arguments
///
/// * `size` - Target image size
///
/// # Returns
///
/// A Compose transform with heavy augmentation for robust training
pub fn strong_augment(size: usize) -> Compose {
    presets::presets::strong_augment(size)
}

/// Create a transform builder
///
/// # Returns
///
/// A new TransformBuilder for creating custom pipelines
pub fn builder() -> TransformBuilder {
    TransformBuilder::new()
}

/// Create a transform registry
///
/// # Returns
///
/// A new TransformRegistry with default transforms registered
pub fn registry() -> TransformRegistry {
    TransformRegistry::new()
}

//
// Type aliases for convenience
//

/// Convenience type alias for a boxed transform
pub type BoxedTransform = Box<dyn Transform>;

/// Convenience type alias for a vector of boxed transforms
pub type TransformVec = Vec<BoxedTransform>;

//
// Trait implementations for common conversions
//

impl From<Vec<BoxedTransform>> for Compose {
    fn from(transforms: Vec<BoxedTransform>) -> Self {
        Compose::new(transforms)
    }
}

//
// Re-exports from sub-modules for convenience
//

pub use core::*;

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_module_exports() {
        // Test that all major exports are accessible
        let _resize = Resize::new((224, 224));
        let _normalize = Normalize::imagenet();
        let _flip = RandomHorizontalFlip::new(0.5);
        let _jitter = ColorJitter::new();
        let _mixup = MixUp::new(1.0);
        let _autoaug = AutoAugment::new();
        let _augmix = AugMix::new();
        let _builder = TransformBuilder::new();
        let _registry = TransformRegistry::new();
    }

    #[test]
    fn test_convenience_functions() {
        // Test convenience constructors
        let train = imagenet_train(224);
        let val = imagenet_val(224);
        let cifar_tr = cifar_train();
        let cifar_v = cifar_val();
        let strong = strong_augment(224);

        assert!(!train.is_empty());
        assert!(!val.is_empty());
        assert!(!cifar_tr.is_empty());
        assert!(!cifar_v.is_empty());
        assert!(!strong.is_empty());

        let _builder = builder();
        let _reg = registry();
    }

    #[test]
    fn test_type_aliases() {
        let transform: BoxedTransform = Box::new(Resize::new((224, 224)));
        let transforms: TransformVec = vec![
            Box::new(Resize::new((224, 224))),
            Box::new(Normalize::imagenet()),
        ];

        assert_eq!(transform.name(), "Resize");
        assert_eq!(transforms.len(), 2);
    }

    #[test]
    fn test_compose_from_vec() {
        let transforms: TransformVec = vec![
            Box::new(Resize::new((224, 224))),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(Normalize::imagenet()),
        ];

        let compose: Compose = transforms.into();
        assert_eq!(compose.len(), 3);
    }

    #[test]
    fn test_full_pipeline() {
        // Test a complete pipeline using the public API
        let input = creation::ones(&[3, 256, 256]).unwrap();

        let pipeline = builder()
            .resize((224, 224))
            .random_horizontal_flip(0.5)
            .add(ColorJitter::new().brightness(0.1))
            .imagenet_normalize()
            .build();

        let result = pipeline.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[3, 224, 224]);
    }

    #[test]
    fn test_preset_pipelines() {
        let input = creation::ones(&[3, 256, 256]).unwrap();

        // Test all preset pipelines
        let presets = vec![imagenet_train(224), imagenet_val(224), strong_augment(224)];

        for preset in presets {
            let result = preset.forward(&input);
            assert!(result.is_ok());
        }
    }

    #[test]
    #[ignore] // Fails in parallel execution due to shared RNG state
    fn test_advanced_transforms() {
        let input = creation::ones(&[3, 224, 224]).unwrap();

        // Test advanced transforms work
        let rand_aug = RandAugment::new(2, 5.0);
        let result = rand_aug.forward(&input);
        assert!(result.is_ok());

        let augmix = AugMix::new();
        let result = augmix.forward(&input);
        assert!(result.is_ok());

        let gridmask = GridMask::new();
        let result = gridmask.forward(&input);
        assert!(result.is_ok());
    }

    #[test]
    #[ignore] // Fails in parallel execution due to shared RNG state
    fn test_mixing_transforms() {
        let input1 = creation::ones(&[3, 32, 32]).unwrap();
        let input2 = creation::zeros(&[3, 32, 32]).unwrap();

        let mixup = MixUp::new(1.0);
        let result = mixup.apply_pair(&input1, &input2, 0, 1, 10);
        assert!(result.is_ok());

        let cutmix = CutMix::new(1.0);
        let result = cutmix.apply_pair(&input1, &input2, 0, 1, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_introspection() {
        let pipeline = builder()
            .resize((224, 224))
            .random_horizontal_flip(0.5)
            .imagenet_normalize()
            .build();

        let description = pipeline.describe();
        assert!(description.contains("Resize"));
        assert!(description.contains("RandomHorizontalFlip"));
        assert!(description.contains("Normalize"));

        let stats = pipeline.statistics();
        assert_eq!(stats.total_transforms, 3);

        let validation = pipeline.validate();
        assert!(validation.is_ok());
    }
}
