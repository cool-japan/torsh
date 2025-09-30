//! Comprehensive Data Augmentation Framework Demo
//!
//! This example demonstrates the complete data augmentation capabilities in ToRSh including:
//! - Basic geometric transforms (resize, crop, flip)
//! - Color augmentations (color jitter, brightness, contrast)
//! - Advanced transforms (random resized crop, random erasing)
//! - Normalization and preprocessing pipelines
//! - Integration with data loading and training workflows

use image::{DynamicImage, ImageBuffer, Rgb};
use std::error::Error;
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;
use torsh_vision::transforms::*;

/// Configuration for data augmentation demonstration
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    pub image_size: (usize, usize),
    pub crop_size: (usize, usize),
    pub batch_size: usize,
    pub num_examples: usize,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            image_size: (256, 256),
            crop_size: (224, 224),
            batch_size: 4,
            num_examples: 8,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🖼️  Comprehensive Data Augmentation Framework Demo");
    println!("==================================================\n");

    let config = AugmentationConfig::default();

    // Demonstrate different augmentation pipelines
    demonstrate_basic_transforms(&config)?;
    demonstrate_training_augmentations(&config)?;
    demonstrate_test_augmentations(&config)?;
    demonstrate_advanced_augmentations(&config)?;
    demonstrate_custom_pipelines(&config)?;
    demonstrate_performance_considerations(&config)?;

    println!("\n✅ Data augmentation framework demonstration completed!");
    Ok(())
}

/// Demonstrate basic geometric transforms
fn demonstrate_basic_transforms(config: &AugmentationConfig) -> Result<(), Box<dyn Error>> {
    println!("🔄 Basic Geometric Transforms Demo");
    println!("==================================\n");

    // Create a sample tensor (C, H, W format)
    let input_tensor = randn(&[3, config.image_size.1, config.image_size.0]);
    println!("📊 Input tensor shape: {:?}", input_tensor.shape());

    // Test basic transforms
    let transforms = vec![
        (
            "Resize",
            Box::new(Resize::new(config.crop_size)) as Box<dyn Transform>,
        ),
        ("Center Crop", Box::new(CenterCrop::new(config.crop_size))),
        (
            "Random Horizontal Flip",
            Box::new(RandomHorizontalFlip::new(0.5)),
        ),
        (
            "Random Vertical Flip",
            Box::new(RandomVerticalFlip::new(0.3)),
        ),
        ("Random Crop", Box::new(RandomCrop::new(config.crop_size))),
    ];

    for (name, transform) in transforms {
        match transform.forward(&input_tensor) {
            Ok(output) => {
                println!(
                    "   ✓ {}: {:?} -> {:?}",
                    name,
                    input_tensor.shape(),
                    output.shape()
                );
            }
            Err(e) => {
                println!("   ✗ {}: Error - {}", name, e);
            }
        }
    }

    println!("\n💡 Basic Transform Guidelines:");
    println!("   • Resize: Use bilinear interpolation for best quality");
    println!("   • Center Crop: Deterministic cropping for test/validation");
    println!("   • Random Crop: Data augmentation for training");
    println!("   • Horizontal Flip: Common for natural images (probability 0.5)");
    println!("   • Vertical Flip: Use carefully, may not be appropriate for all datasets\n");

    Ok(())
}

/// Demonstrate training augmentation pipeline
fn demonstrate_training_augmentations(config: &AugmentationConfig) -> Result<(), Box<dyn Error>> {
    println!("🏋️  Training Augmentation Pipeline Demo");
    println!("======================================\n");

    let input_tensor = randn(&[3, config.image_size.1, config.image_size.0]);

    // Training augmentation pipeline (more aggressive)
    let training_pipeline = Compose::new(vec![
        Box::new(
            RandomResizedCrop::new(config.crop_size)
                .with_scale((0.08, 1.0))
                .with_ratio((3.0 / 4.0, 4.0 / 3.0)),
        ),
        Box::new(RandomHorizontalFlip::new(0.5)),
        Box::new(
            ColorJitter::new()
                .brightness(0.4)
                .contrast(0.4)
                .saturation(0.4)
                .hue(0.1),
        ),
        Box::new(
            RandomErasing::new(0.25)
                .with_scale((0.02, 0.33))
                .with_ratio((0.3, 3.3)),
        ),
        Box::new(Normalize::new(
            vec![0.485, 0.456, 0.406], // ImageNet mean
            vec![0.229, 0.224, 0.225], // ImageNet std
        )),
    ]);

    println!("📈 Training Pipeline Components:");
    println!("   1. Random Resized Crop (scale: 0.08-1.0, ratio: 0.75-1.33)");
    println!("   2. Random Horizontal Flip (p=0.5)");
    println!("   3. Color Jitter (brightness±0.4, contrast±0.4, saturation±0.4, hue±0.1)");
    println!("   4. Random Erasing (p=0.25, scale: 0.02-0.33)");
    println!("   5. ImageNet Normalization");

    // Apply pipeline multiple times to show variation
    println!("\n🎲 Pipeline Variation Examples:");
    for i in 1..=3 {
        match training_pipeline.forward(&input_tensor) {
            Ok(output) => {
                println!(
                    "   Example {}: {:?} -> {:?}",
                    i,
                    input_tensor.shape(),
                    output.shape()
                );
            }
            Err(e) => {
                println!("   Example {}: Error - {}", i, e);
            }
        }
    }

    println!("\n💡 Training Augmentation Best Practices:");
    println!("   • Use random resized crop for scale and aspect ratio invariance");
    println!("   • Apply horizontal flip for natural images");
    println!("   • Use moderate color jitter to improve robustness");
    println!("   • Random erasing helps with occlusion robustness");
    println!("   • Always normalize as the final step\n");

    Ok(())
}

/// Demonstrate test/validation augmentation pipeline
fn demonstrate_test_augmentations(config: &AugmentationConfig) -> Result<(), Box<dyn Error>> {
    println!("🧪 Test/Validation Augmentation Pipeline Demo");
    println!("==============================================\n");

    let input_tensor = randn(&[3, config.image_size.1, config.image_size.0]);

    // Test augmentation pipeline (deterministic)
    let test_pipeline = Compose::new(vec![
        Box::new(Resize::new((256, 256))),
        Box::new(CenterCrop::new(config.crop_size)),
        Box::new(Normalize::new(
            vec![0.485, 0.456, 0.406], // ImageNet mean
            vec![0.229, 0.224, 0.225], // ImageNet std
        )),
    ]);

    println!("🔬 Test Pipeline Components:");
    println!("   1. Resize to 256x256 (preserve aspect ratio)");
    println!("   2. Center Crop to 224x224");
    println!("   3. ImageNet Normalization");

    match test_pipeline.forward(&input_tensor) {
        Ok(output) => {
            println!(
                "\n✓ Test pipeline result: {:?} -> {:?}",
                input_tensor.shape(),
                output.shape()
            );
        }
        Err(e) => {
            println!("\n✗ Test pipeline error: {}", e);
        }
    }

    // Alternative: Ten Crop for test-time augmentation
    println!("\n📊 Test-Time Augmentation (TTA) Options:");
    println!("   • Single Center Crop (fastest, implemented above)");
    println!("   • Ten Crop: 4 corners + center + horizontal flips");
    println!("   • Multi-Scale Testing: Test at different scales");
    println!("   • Ensemble Methods: Average predictions from multiple augmentations");

    println!("\n💡 Test Augmentation Guidelines:");
    println!("   • Keep deterministic for reproducible results");
    println!("   • Use center crop for fair evaluation");
    println!("   • Consider TTA for improved accuracy (at cost of speed)");
    println!("   • Match normalization used during training\n");

    Ok(())
}

/// Demonstrate advanced augmentation techniques
fn demonstrate_advanced_augmentations(_config: &AugmentationConfig) -> Result<(), Box<dyn Error>> {
    println!("🚀 Advanced Augmentation Techniques Demo");
    println!("========================================\n");

    let input_tensor = randn(&[3, 224, 224]);

    println!("🎯 Advanced Techniques Available:");

    // Random Erasing variations
    println!("\n1. Random Erasing Variants:");
    let erasing_configs = vec![
        (
            "Conservative",
            RandomErasing::new(0.1).with_scale((0.02, 0.15)),
        ),
        (
            "Standard",
            RandomErasing::new(0.25).with_scale((0.02, 0.33)),
        ),
        (
            "Aggressive",
            RandomErasing::new(0.5).with_scale((0.05, 0.5)),
        ),
    ];

    for (name, transform) in erasing_configs {
        match transform.forward(&input_tensor) {
            Ok(_) => println!("   ✓ {} Erasing: Successfully applied", name),
            Err(e) => println!("   ✗ {} Erasing: Error - {}", name, e),
        }
    }

    // Color Jitter variations
    println!("\n2. Color Jitter Variants:");
    let color_configs = vec![
        ("Subtle", ColorJitter::new().brightness(0.1).contrast(0.1)),
        (
            "Moderate",
            ColorJitter::new()
                .brightness(0.2)
                .contrast(0.2)
                .saturation(0.2),
        ),
        (
            "Strong",
            ColorJitter::new()
                .brightness(0.4)
                .contrast(0.4)
                .saturation(0.4)
                .hue(0.1),
        ),
    ];

    for (name, transform) in color_configs {
        match transform.forward(&input_tensor) {
            Ok(_) => println!("   ✓ {} Color Jitter: Successfully applied", name),
            Err(e) => println!("   ✗ {} Color Jitter: Error - {}", name, e),
        }
    }

    // Geometric variations
    println!("\n3. Geometric Transform Variants:");
    let geometric_configs = vec![
        (
            "Mild Crop",
            RandomResizedCrop::new((224, 224)).with_scale((0.8, 1.0)),
        ),
        (
            "Standard Crop",
            RandomResizedCrop::new((224, 224)).with_scale((0.08, 1.0)),
        ),
        (
            "Extreme Crop",
            RandomResizedCrop::new((224, 224)).with_scale((0.05, 1.0)),
        ),
    ];

    for (name, transform) in geometric_configs {
        match transform.forward(&input_tensor) {
            Ok(_) => println!("   ✓ {}: Successfully applied", name),
            Err(e) => println!("   ✗ {}: Error - {}", name, e),
        }
    }

    println!("\n💡 Advanced Augmentation Guidelines:");
    println!("   • Random Erasing: Start conservative, increase for robustness");
    println!("   • Color Jitter: Adjust based on dataset characteristics");
    println!("   • Geometric Transforms: Balance diversity with semantic preservation");
    println!("   • Always validate augmentations don't hurt performance");
    println!("   • Consider domain-specific augmentations for specialized tasks\n");

    Ok(())
}

/// Demonstrate custom augmentation pipelines
fn demonstrate_custom_pipelines(_config: &AugmentationConfig) -> Result<(), Box<dyn Error>> {
    println!("⚙️  Custom Augmentation Pipelines Demo");
    println!("======================================\n");

    let input_tensor = randn(&[3, 224, 224]);

    println!("📋 Task-Specific Pipeline Examples:\n");

    // 1. Classification Pipeline
    println!("1. 🏷️  Image Classification Pipeline:");
    let classification_pipeline = Compose::new(vec![
        Box::new(RandomResizedCrop::new((224, 224))),
        Box::new(RandomHorizontalFlip::new(0.5)),
        Box::new(
            ColorJitter::new()
                .brightness(0.4)
                .contrast(0.4)
                .saturation(0.4),
        ),
        Box::new(Normalize::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        )),
    ]);

    match classification_pipeline.forward(&input_tensor) {
        Ok(_) => println!("   ✓ Classification pipeline: Successfully applied"),
        Err(e) => println!("   ✗ Classification pipeline: Error - {}", e),
    }

    // 2. Object Detection Pipeline (more conservative)
    println!("\n2. 📦 Object Detection Pipeline:");
    let detection_pipeline = Compose::new(vec![
        Box::new(Resize::new((512, 512))),
        Box::new(RandomHorizontalFlip::new(0.5)),
        Box::new(ColorJitter::new().brightness(0.2).contrast(0.2)),
        Box::new(Normalize::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        )),
    ]);

    match detection_pipeline.forward(&input_tensor) {
        Ok(_) => println!("   ✓ Detection pipeline: Successfully applied"),
        Err(e) => println!("   ✗ Detection pipeline: Error - {}", e),
    }

    // 3. Self-supervised Learning Pipeline (aggressive)
    println!("\n3. 🔄 Self-Supervised Learning Pipeline:");
    let ssl_pipeline = Compose::new(vec![
        Box::new(RandomResizedCrop::new((224, 224)).with_scale((0.2, 1.0))),
        Box::new(RandomHorizontalFlip::new(0.5)),
        Box::new(
            ColorJitter::new()
                .brightness(0.8)
                .contrast(0.8)
                .saturation(0.8)
                .hue(0.2),
        ),
        Box::new(RandomErasing::new(0.5)),
        Box::new(Normalize::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        )),
    ]);

    match ssl_pipeline.forward(&input_tensor) {
        Ok(_) => println!("   ✓ Self-supervised pipeline: Successfully applied"),
        Err(e) => println!("   ✗ Self-supervised pipeline: Error - {}", e),
    }

    // 4. Fine-tuning Pipeline (conservative)
    println!("\n4. 🎯 Fine-tuning Pipeline:");
    let finetune_pipeline = Compose::new(vec![
        Box::new(Resize::new((256, 256))),
        Box::new(RandomCrop::new((224, 224))),
        Box::new(RandomHorizontalFlip::new(0.5)),
        Box::new(ColorJitter::new().brightness(0.1).contrast(0.1)),
        Box::new(Normalize::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        )),
    ]);

    match finetune_pipeline.forward(&input_tensor) {
        Ok(_) => println!("   ✓ Fine-tuning pipeline: Successfully applied"),
        Err(e) => println!("   ✗ Fine-tuning pipeline: Error - {}", e),
    }

    println!("\n💡 Custom Pipeline Design Principles:");
    println!("   • Classification: Aggressive augmentation for generalization");
    println!("   • Object Detection: Preserve spatial relationships");
    println!("   • Self-Supervised: Very aggressive to learn robust features");
    println!("   • Fine-tuning: Conservative to preserve pre-trained features");
    println!("   • Always validate on your specific dataset and task\n");

    Ok(())
}

/// Demonstrate performance considerations and optimizations
fn demonstrate_performance_considerations(
    _config: &AugmentationConfig,
) -> Result<(), Box<dyn Error>> {
    println!("⚡ Performance Considerations Demo");
    println!("==================================\n");

    println!("🚀 Optimization Strategies:\n");

    println!("1. 🔄 Transform Ordering:");
    println!("   ✓ Good: Crop → Flip → Color → Normalize");
    println!("   ✗ Bad: Resize → Crop → Resize (redundant operations)");
    println!("   • Apply spatial transforms before color transforms");
    println!("   • Keep expensive operations (resize) to minimum");

    println!("\n2. 💾 Memory Efficiency:");
    println!("   • Use in-place operations where possible");
    println!("   • Avoid creating unnecessary tensor copies");
    println!("   • Consider tensor slicing over full copies for crops");

    println!("\n3. 🔢 Batch Processing:");
    println!("   • Apply transforms to entire batches when possible");
    println!("   • Use vectorized operations for normalization");
    println!("   • Leverage GPU acceleration for compute-intensive transforms");

    println!("\n4. 🎲 Randomization Strategy:");
    println!("   • Use seeded random generators for reproducibility");
    println!("   • Pre-compute random parameters for entire epochs");
    println!("   • Balance randomness with performance requirements");

    println!("\n5. 📊 Monitoring and Profiling:");
    println!("   • Profile transform pipelines to identify bottlenecks");
    println!("   • Monitor GPU/CPU utilization during data loading");
    println!("   • Use async data loading to overlap computation");

    println!("\n💡 Performance Best Practices:");
    println!("   • Profile your specific pipeline on target hardware");
    println!("   • Use appropriate tensor backends (CPU vs GPU)");
    println!("   • Consider caching transformed data for small datasets");
    println!("   • Implement custom transforms for domain-specific needs");
    println!("   • Balance augmentation strength with training speed\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augmentation_pipeline() {
        let config = AugmentationConfig {
            image_size: (64, 64),
            crop_size: (32, 32),
            batch_size: 2,
            num_examples: 4,
        };

        // Test that basic pipeline runs without errors
        assert!(demonstrate_basic_transforms(&config).is_ok());
        assert!(demonstrate_training_augmentations(&config).is_ok());
        assert!(demonstrate_test_augmentations(&config).is_ok());
    }

    #[test]
    fn test_transform_composition() {
        let input = randn(&[3, 64, 64]);

        let pipeline = Compose::new(vec![
            Box::new(Resize::new((32, 32))),
            Box::new(RandomHorizontalFlip::new(0.5)),
            Box::new(Normalize::new(vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5])),
        ]);

        let result = pipeline.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[3, 32, 32]);
    }

    #[test]
    fn test_individual_transforms() {
        let input = randn(&[3, 64, 64]);

        // Test individual transforms
        assert!(Resize::new((32, 32)).forward(&input).is_ok());
        assert!(RandomHorizontalFlip::new(0.5).forward(&input).is_ok());
        assert!(RandomVerticalFlip::new(0.5).forward(&input).is_ok());
        assert!(ColorJitter::new().brightness(0.2).forward(&input).is_ok());
        assert!(RandomErasing::new(0.1).forward(&input).is_ok());
    }
}
