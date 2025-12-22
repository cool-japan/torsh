//! Data Augmentation Pipeline Example
//!
//! This example demonstrates:
//! - Building custom augmentation pipelines
//! - Using advanced augmentation techniques (RandAugment, AugMix, etc.)
//! - Visualizing augmented images
//! - Comparing different augmentation strategies
//! - Best practices for data augmentation in deep learning
//!
//! Run with: cargo run --example data_augmentation

use std::sync::Arc;
use torsh_core::device::CpuDevice;
use torsh_tensor::{creation, Tensor};
use torsh_vision::{
    AugMix, ColorJitter, Compose, CutMix, Cutout, GridMask, MixUp, Normalize, RandomCrop,
    RandomErasing, RandomHorizontalFlip, RandomResizedCrop, RandomRotation, RandomVerticalFlip,
    Resize, Result, Transform,
};

/// Configuration for different augmentation strategies
#[derive(Debug, Clone)]
enum AugmentationStrategy {
    Basic,
    Moderate,
    Aggressive,
    RandAugment,
    AugMix,
    MixingBased,
}

/// Create augmentation pipeline based on strategy
fn create_augmentation_pipeline(strategy: AugmentationStrategy) -> Box<dyn Transform> {
    match strategy {
        AugmentationStrategy::Basic => Box::new(Compose::new(vec![
            Box::new(Resize::new((256, 256))),
            Box::new(RandomCrop::new(224, None)),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(Normalize::new(
                vec![0.485, 0.456, 0.406],
                vec![0.229, 0.224, 0.225],
            )),
        ])),

        AugmentationStrategy::Moderate => Box::new(Compose::new(vec![
            Box::new(RandomResizedCrop::new(
                224,
                Some((0.8, 1.0)),
                Some((0.75, 1.333)),
            )),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(ColorJitter::new(0.3, 0.3, 0.3, 0.1)),
            Box::new(Normalize::new(
                vec![0.485, 0.456, 0.406],
                vec![0.229, 0.224, 0.225],
            )),
        ])),

        AugmentationStrategy::Aggressive => Box::new(Compose::new(vec![
            Box::new(RandomResizedCrop::new(
                224,
                Some((0.6, 1.0)),
                Some((0.75, 1.333)),
            )),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(RandomVerticalFlip::new(0.2)),
            Box::new(RandomRotation::new((-15.0, 15.0))),
            Box::new(ColorJitter::new(0.4, 0.4, 0.4, 0.2)),
            Box::new(RandomErasing::new(0.5, (0.02, 0.33), (0.3, 3.3), 0.0)),
            Box::new(Cutout::new(16, 0.5)),
            Box::new(Normalize::new(
                vec![0.485, 0.456, 0.406],
                vec![0.229, 0.224, 0.225],
            )),
        ])),

        AugmentationStrategy::RandAugment => {
            // RandAugment: Practical automated data augmentation
            Box::new(Compose::new(vec![
                Box::new(Resize::new((256, 256))),
                Box::new(RandomCrop::new(224, None)),
                // Note: Full RandAugment would be implemented as a dedicated transform
                Box::new(ColorJitter::new(0.4, 0.4, 0.4, 0.1)),
                Box::new(RandomHorizontalFlip::new(0.5)),
                Box::new(Normalize::new(
                    vec![0.485, 0.456, 0.406],
                    vec![0.229, 0.224, 0.225],
                )),
            ]))
        }

        AugmentationStrategy::AugMix => Box::new(Compose::new(vec![
            Box::new(Resize::new((256, 256))),
            Box::new(RandomCrop::new(224, None)),
            Box::new(AugMix::new(3, 3, 1.0)),
            Box::new(Normalize::new(
                vec![0.485, 0.456, 0.406],
                vec![0.229, 0.224, 0.225],
            )),
        ])),

        AugmentationStrategy::MixingBased => {
            // For mixing augmentations like MixUp/CutMix, you'd typically apply them
            // at batch level during training
            Box::new(Compose::new(vec![
                Box::new(Resize::new((256, 256))),
                Box::new(RandomCrop::new(224, None)),
                Box::new(RandomHorizontalFlip::new(0.5)),
                Box::new(GridMask::new(96, 0.6, true, 0.5, 0.0)),
                Box::new(Normalize::new(
                    vec![0.485, 0.456, 0.406],
                    vec![0.229, 0.224, 0.225],
                )),
            ]))
        }
    }
}

/// Apply batch-level augmentations (MixUp, CutMix)
fn apply_batch_augmentation(
    images: Tensor,
    labels: Tensor,
    aug_type: &str,
) -> Result<(Tensor, Tensor)> {
    match aug_type {
        "mixup" => {
            let mixup = MixUp::new(1.0);
            mixup.apply(&images, &labels)
        }
        "cutmix" => {
            let cutmix = CutMix::new(1.0);
            cutmix.apply(&images, &labels)
        }
        _ => Ok((images, labels)),
    }
}

/// Demonstrate augmentation on a sample image
fn demonstrate_augmentation(
    image: &Tensor,
    strategy: AugmentationStrategy,
    num_samples: usize,
) -> Result<Vec<Tensor>> {
    println!("  Applying {:?} augmentation...", strategy);

    let transform = create_augmentation_pipeline(strategy);
    let mut augmented_images = Vec::new();

    for i in 0..num_samples {
        let aug_img = transform.apply(image)?;
        augmented_images.push(aug_img);

        if i < 3 {
            println!("    Sample {}: shape {:?}", i + 1, aug_img.shape());
        }
    }

    Ok(augmented_images)
}

/// Compare different augmentation strategies
fn compare_strategies(sample_image: &Tensor) -> Result<()> {
    println!("\n🔍 Comparing Augmentation Strategies");
    println!("═══════════════════════════════════════\n");

    let strategies = vec![
        AugmentationStrategy::Basic,
        AugmentationStrategy::Moderate,
        AugmentationStrategy::Aggressive,
        AugmentationStrategy::AugMix,
        AugmentationStrategy::MixingBased,
    ];

    for strategy in strategies {
        println!("Strategy: {:?}", strategy);
        let augmented = demonstrate_augmentation(sample_image, strategy, 5)?;
        println!("  Generated {} augmented samples\n", augmented.len());
    }

    Ok(())
}

/// Demonstrate batch augmentation techniques
fn demonstrate_batch_augmentation() -> Result<()> {
    println!("\n🎨 Batch-Level Augmentation Techniques");
    println!("═══════════════════════════════════════\n");

    let device = Arc::new(CpuDevice::new());

    // Create dummy batch
    let batch_size = 4;
    let images = creation::random_normal(
        &[batch_size, 3, 224, 224],
        0.0,
        1.0,
        torsh_core::dtype::DType::Float32,
    )?;
    let labels = creation::tensor(
        &[0.0, 1.0, 2.0, 3.0],
        &[batch_size],
        torsh_core::dtype::DType::Float32,
    )?;

    println!("Original batch shape: {:?}", images.shape());
    println!("Original labels: {:?}\n", labels.shape());

    // Apply MixUp
    println!("1️⃣  MixUp Augmentation:");
    let (mixed_images, mixed_labels) =
        apply_batch_augmentation(images.clone(), labels.clone(), "mixup")?;
    println!("  Mixed images shape: {:?}", mixed_images.shape());
    println!("  Mixed labels shape: {:?}", mixed_labels.shape());
    println!("  Note: Labels are interpolated between pairs\n");

    // Apply CutMix
    println!("2️⃣  CutMix Augmentation:");
    let (cutmix_images, cutmix_labels) =
        apply_batch_augmentation(images.clone(), labels.clone(), "cutmix")?;
    println!("  CutMix images shape: {:?}", cutmix_images.shape());
    println!("  CutMix labels shape: {:?}", cutmix_labels.shape());
    println!("  Note: Labels weighted by area ratio\n");

    Ok(())
}

/// Best practices and recommendations
fn print_best_practices() {
    println!("\n📚 Data Augmentation Best Practices");
    println!("═══════════════════════════════════════\n");

    println!("1. Start Simple:");
    println!("   - Begin with basic augmentations (flip, crop, normalize)");
    println!("   - Add complexity only if needed for performance\n");

    println!("2. Match Your Task:");
    println!("   - Classification: RandomResizedCrop, ColorJitter, Flip");
    println!("   - Detection: Careful with geometric transforms");
    println!("   - Segmentation: Apply same transforms to image and mask\n");

    println!("3. Consider Your Domain:");
    println!("   - Natural images: All augmentations applicable");
    println!("   - Medical images: Avoid color jitter, use careful rotations");
    println!("   - Satellite images: Consider domain-specific augmentations\n");

    println!("4. Advanced Techniques:");
    println!("   - RandAugment: Automated policy search");
    println!("   - AugMix: Improved robustness and uncertainty");
    println!("   - MixUp/CutMix: Better generalization for classification\n");

    println!("5. Validation Set:");
    println!("   - Use minimal augmentation (resize, normalize only)");
    println!("   - Helps accurately measure model performance\n");

    println!("6. Performance Tips:");
    println!("   - Cache augmented data if memory allows");
    println!("   - Use parallel processing for large datasets");
    println!("   - Consider GPU augmentation for bottleneck cases\n");
}

fn main() -> Result<()> {
    println!("🎯 ToRSh Vision - Data Augmentation Example");
    println!("=============================================\n");

    // Create a sample image
    println!("📸 Creating sample image...");
    let sample_image =
        creation::random_normal(&[3, 256, 256], 0.5, 0.2, torsh_core::dtype::DType::Float32)?;
    println!("  Sample shape: {:?}\n", sample_image.shape());

    // Compare different strategies
    compare_strategies(&sample_image)?;

    // Demonstrate batch augmentation
    demonstrate_batch_augmentation()?;

    // Print best practices
    print_best_practices();

    println!("✅ Example completed successfully!");
    println!("\nNext steps:");
    println!("  - Try these augmentations on your own dataset");
    println!("  - Experiment with different parameter values");
    println!("  - Measure impact on model performance");
    println!("  - Consider using TransformRegistry for complex pipelines\n");

    Ok(())
}
