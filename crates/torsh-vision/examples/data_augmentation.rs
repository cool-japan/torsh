//! Data Augmentation Pipeline Example
//!
//! This example demonstrates:
//! - Building custom augmentation pipelines
//! - Using advanced augmentation techniques (RandAugment, AugMix, etc.)
//! - Comparing different augmentation strategies
//! - Best practices for data augmentation in deep learning
//!
//! Run with: cargo run --example data_augmentation

use std::sync::Arc;
use torsh_core::device::CpuDevice;
use torsh_tensor::{creation, Tensor};
use torsh_vision::{
    ColorJitter, CutMix, Cutout, MixUp, Normalize, RandomCrop, RandomErasing, RandomHorizontalFlip,
    RandomResizedCrop, RandomRotation, RandomVerticalFlip, Resize, Result,
};

/// Configuration for different augmentation strategies
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum AugmentationStrategy {
    Basic,
    Moderate,
    Aggressive,
}

fn main() -> Result<()> {
    println!("🎨 ToRSh Vision - Data Augmentation Example");
    println!("============================================\n");

    let _device = Arc::new(CpuDevice::new());

    // Create sample image (3 channels, 256x256)
    println!("📸 Creating sample image...");
    let sample_image: Tensor<f32> = creation::randn(&[3, 256, 256])?;
    println!("  Image shape: {:?}\n", sample_image.shape());

    // Demonstrate individual transforms
    println!("🔧 Individual Transforms:");
    println!("════════════════════════\n");

    // Resize
    println!("1️⃣  Resize:");
    let _resize = Resize::new((224, 224));
    println!("  Resizes image to 224x224\n");

    // Random Crop
    println!("2️⃣  RandomCrop:");
    let _random_crop = RandomCrop::new((224, 224));
    println!("  Randomly crops 224x224 patch from image\n");

    // Random Horizontal Flip
    println!("3️⃣  RandomHorizontalFlip:");
    let _hflip = RandomHorizontalFlip::new(0.5);
    println!("  Flips image horizontally with 50% probability\n");

    // Random Vertical Flip
    println!("4️⃣  RandomVerticalFlip:");
    let _vflip = RandomVerticalFlip::new(0.2);
    println!("  Flips image vertically with 20% probability\n");

    // Random Rotation
    println!("5️⃣  RandomRotation:");
    let _rotation = RandomRotation::new((-15.0, 15.0));
    println!("  Rotates image randomly between -15 and +15 degrees\n");

    // Color Jitter
    println!("6️⃣  ColorJitter:");
    let _color_jitter = ColorJitter::new()
        .brightness(0.3)
        .contrast(0.3)
        .saturation(0.3)
        .hue(0.1);
    println!("  Adjusts brightness, contrast, saturation, hue\n");

    // Random Erasing
    println!("7️⃣  RandomErasing:");
    let _erasing = RandomErasing::new(0.5)
        .with_scale((0.02, 0.33))
        .with_ratio((0.3, 3.3))
        .with_value(0.0);
    println!("  Randomly erases rectangular region\n");

    // Cutout
    println!("8️⃣  Cutout:");
    let _cutout = Cutout::new(16, 2); // length=16, n_holes=2
    println!("  Removes square patches from image\n");

    // Normalize
    println!("9️⃣  Normalize:");
    let _normalize = Normalize::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225]);
    println!("  Normalizes with ImageNet mean/std\n");

    // Random Resized Crop
    println!("🔟 RandomResizedCrop:");
    let _rrc = RandomResizedCrop::new((224, 224))
        .with_scale((0.8, 1.0))
        .with_ratio((0.75, 1.333));
    println!("  Crops and resizes to 224x224\n");

    // Demonstrate Mixing Augmentations
    println!("🔀 Mixing Augmentations:");
    println!("════════════════════════\n");

    println!("1️⃣  MixUp:");
    let _mixup = MixUp::new(1.0);
    println!("  Blends two images and their labels");
    println!("  Use apply_pair() with two images and labels\n");

    println!("2️⃣  CutMix:");
    let _cutmix = CutMix::new(1.0);
    println!("  Cuts and pastes patches between images");
    println!("  Use apply_pair() with two images and labels\n");

    // Augmentation Strategies
    println!("📋 Augmentation Strategies:");
    println!("═══════════════════════════\n");

    println!("Basic Strategy:");
    println!("  - Resize → RandomCrop → RandomHorizontalFlip → Normalize");
    println!("  - Best for: Standard image classification\n");

    println!("Moderate Strategy:");
    println!("  - RandomResizedCrop → RandomHorizontalFlip → ColorJitter → Normalize");
    println!("  - Best for: General purpose training\n");

    println!("Aggressive Strategy:");
    println!("  - RandomResizedCrop → Flips → Rotation → ColorJitter → RandomErasing → Cutout → Normalize");
    println!("  - Best for: When you have limited data\n");

    // Best practices
    println!("📚 Data Augmentation Best Practices:");
    println!("═════════════════════════════════════\n");

    println!("1. Start Simple:");
    println!("   - Begin with basic augmentations (flip, crop)");
    println!("   - Add complexity gradually based on validation performance\n");

    println!("2. Match Domain:");
    println!("   - Medical imaging: Be careful with flips");
    println!("   - Satellite: Use all rotations");
    println!("   - Text: Avoid geometric transforms\n");

    println!("3. Preserve Labels:");
    println!("   - Some augmentations may change the correct label");
    println!("   - Verify augmentations don't break classification\n");

    println!("4. Use Mixing at Batch Level:");
    println!("   - Apply MixUp/CutMix during batch loading");
    println!("   - Helps regularization and reduces overfitting\n");

    println!("5. AutoAugment/RandAugment:");
    println!("   - Use learned or random policies for best results");
    println!("   - RandAugment: Simpler and often as effective\n");

    println!("✅ Example completed successfully!");
    println!("\nNext steps:");
    println!("  - Integrate transforms into training pipeline");
    println!("  - Experiment with different strategies");
    println!("  - Monitor validation performance\n");

    Ok(())
}
