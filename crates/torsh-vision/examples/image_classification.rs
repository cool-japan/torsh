//! Image Classification Example
//!
//! This example demonstrates:
//! - Basic CNN architecture concepts
//! - Image preprocessing with ToRSh Vision
//! - Model structure for classification tasks
//!
//! Run with: cargo run --example image_classification

use std::sync::Arc;
use torsh_core::device::CpuDevice;
use torsh_tensor::{creation, Tensor};
use torsh_vision::{Normalize, RandomHorizontalFlip, Resize, Result};

fn main() -> Result<()> {
    println!("🖼️  ToRSh Vision - Image Classification Example");
    println!("================================================\n");

    let _device = Arc::new(CpuDevice::new());

    // Configuration
    println!("📊 Configuration:");
    println!("  Image size: 32x32 (CIFAR-10)");
    println!("  Number of classes: 10");
    println!("  Batch size: 32\n");

    // Create sample batch
    println!("📸 Creating sample batch...");
    let batch_size = 32;
    let channels = 3;
    let height = 32;
    let width = 32;

    let sample_batch: Tensor<f32> = creation::randn(&[batch_size, channels, height, width])?;
    println!("  Batch shape: {:?}\n", sample_batch.shape());

    // Demonstrate transforms
    println!("🔧 Image Transforms:");
    println!("═══════════════════\n");

    println!("1️⃣  Resize:");
    let _resize = Resize::new((224, 224));
    println!("  Resize images to 224x224 for ImageNet models\n");

    println!("2️⃣  Random Horizontal Flip:");
    let _hflip = RandomHorizontalFlip::new(0.5);
    println!("  Randomly flip images with 50% probability\n");

    println!("3️⃣  Normalize:");
    let _normalize = Normalize::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225]);
    println!("  Normalize with ImageNet mean and std\n");

    // CNN Architecture
    println!("🏗️  CNN Architecture for CIFAR-10:");
    println!("═══════════════════════════════════\n");

    println!("Feature Extractor:");
    println!("  Conv2d(3 → 32, 3×3, padding=1) + ReLU");
    println!("  Conv2d(32 → 64, 3×3, padding=1) + ReLU");
    println!("  MaxPool2d(2×2)");
    println!("  Conv2d(64 → 128, 3×3, padding=1) + ReLU");
    println!("  MaxPool2d(2×2)\n");

    println!("Classifier:");
    println!("  Flatten");
    println!("  Linear(128×8×8 → 256) + ReLU");
    println!("  Linear(256 → 10)\n");

    // Training loop structure
    println!("📚 Training Loop Structure:");
    println!("═══════════════════════════\n");

    println!("for epoch in 0..epochs {{");
    println!("    model.set_training(true);");
    println!("    for (images, labels) in train_loader {{");
    println!("        optimizer.zero_grad();");
    println!("        let outputs = model.forward(&images);");
    println!("        let loss = cross_entropy(&outputs, &labels);");
    println!("        loss.backward();");
    println!("        optimizer.step();");
    println!("    }}");
    println!("    // Evaluate on validation set");
    println!("}}\n");

    // Common optimizers
    println!("⚙️  Common Optimizers:");
    println!("═══════════════════════\n");

    println!("1. SGD with momentum:");
    println!("   SGD::new(params, 0.01, 0.9)");
    println!();

    println!("2. Adam:");
    println!("   Adam::new(params, 0.001, (0.9, 0.999))");
    println!();

    println!("3. AdamW (with weight decay):");
    println!("   AdamW::new(params, 0.001, 0.01)");
    println!();

    // Best practices
    println!("📖 Best Practices:");
    println!("══════════════════\n");

    println!("1. Data Augmentation:");
    println!("   - Random crop with padding");
    println!("   - Random horizontal flip");
    println!("   - Color jitter for robustness\n");

    println!("2. Learning Rate Schedule:");
    println!("   - Start with warm-up");
    println!("   - Cosine annealing");
    println!("   - Step decay at milestones\n");

    println!("3. Regularization:");
    println!("   - Dropout in classifier");
    println!("   - Weight decay (L2 regularization)");
    println!("   - Label smoothing\n");

    println!("4. Batch Normalization:");
    println!("   - Add after each conv layer");
    println!("   - Helps with training stability\n");

    println!("✅ Example completed successfully!");
    println!("\nNext steps:");
    println!("  - Load actual CIFAR-10 dataset");
    println!("  - Implement training loop");
    println!("  - Add validation and checkpointing");
    println!("  - Experiment with architectures\n");

    Ok(())
}
