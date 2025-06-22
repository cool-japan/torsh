//! ResNet inference example using ToRSh model zoo
//! 
//! This example demonstrates:
//! - Loading a ResNet model from the model zoo
//! - Image preprocessing
//! - Model inference
//! - Results interpretation

use torsh::prelude::*;

#[cfg(feature = "model-zoo")]
use torsh_models::{
    vision::{ResNet, VisionModelUtils, ImagePreprocessor},
    prelude::*
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 ToRSh ResNet Inference Example");
    println!("==================================");
    
    // Check if model zoo is available
    #[cfg(not(feature = "model-zoo"))]
    {
        println!("❌ Model zoo feature not enabled. Please compile with --features model-zoo");
        return Ok(());
    }
    
    #[cfg(feature = "model-zoo")]
    {
        // Create ResNet-18 model
        println!("🏗️  Creating ResNet-18 model...");
        let mut model = ResNet::resnet18(1000); // ImageNet classes
        model.eval();
        
        // Get model info
        if let Some(variant) = VisionModelUtils::get_model_variant("resnet18") {
            println!("Model: {}", variant.variant);
            println!("Architecture: {:?}", variant.architecture);
            println!("Parameters: {:.1}M", variant.parameters as f32 / 1_000_000.0);
            println!("Input size: {:?}", variant.input_size);
            if let Some(acc) = variant.imagenet_top1_accuracy {
                println!("ImageNet Top-1: {:.2}%", acc);
            }
        }
        
        // Create preprocessor
        println!("\\n🖼️  Setting up image preprocessing...");
        let preprocessor = ImagePreprocessor::imagenet();
        println!("Target size: {:?}", preprocessor.target_size);
        println!("Normalization mean: {:?}", preprocessor.mean);
        println!("Normalization std: {:?}", preprocessor.std);
        
        // Simulate image data (normally you'd load an actual image)
        println!("\\n📸 Creating synthetic image data...");
        let batch_size = 1;
        let (channels, height, width) = (3, 224, 224);
        
        // Create random image tensor (normally preprocessed real image)
        let input_image = Tensor::randn(&[batch_size, channels, height, width])?;
        println!("Input shape: {:?}", input_image.shape());
        
        // Model inference
        println!("\\n🚀 Running inference...");
        let start_time = std::time::Instant::now();
        
        let logits = model.forward(&input_image)?;
        
        let inference_time = start_time.elapsed();
        println!("Inference time: {:.2}ms", inference_time.as_millis());
        println!("Output shape: {:?}", logits.shape());
        
        // Convert logits to probabilities
        println!("\\n📊 Processing results...");
        let logits_data: Vec<f32> = logits.to_vec()?;
        let probabilities = VisionModelUtils::softmax(&logits_data);
        
        // Get top-5 predictions
        let imagenet_classes = get_sample_imagenet_classes();
        let top_5 = VisionModelUtils::get_top_k_predictions(
            &probabilities, 
            5, 
            Some(&imagenet_classes)
        );
        
        println!("\\n🏆 Top-5 Predictions:");
        for (i, (class_idx, confidence, class_name)) in top_5.iter().enumerate() {
            println!(
                "  {}. {}: {:.2}% (class {})",
                i + 1,
                class_name.as_ref().unwrap_or(&format!("Class {}", class_idx)),
                confidence * 100.0,
                class_idx
            );
        }
        
        // Performance metrics
        println!("\\n⚡ Performance Metrics:");
        println!("  Throughput: {:.1} FPS", 1000.0 / inference_time.as_millis() as f32);
        println!("  Memory usage: ~{:.1} MB", estimate_memory_usage(batch_size, channels, height, width));
        
        // Model statistics
        println!("\\n📈 Model Statistics:");
        println!("  Parameters: {:.1}M", 11.7); // ResNet-18 has ~11.7M parameters
        println!("  FLOPs: ~1.8 GFLOPs"); // Approximate for 224x224 input
        println!("  Model size: ~45 MB"); // Approximate model size
    }
    
    Ok(())
}

/// Get sample ImageNet class names (first 10 classes)
fn get_sample_imagenet_classes() -> Vec<String> {
    vec![
        "tench".to_string(),
        "goldfish".to_string(),
        "great white shark".to_string(),
        "tiger shark".to_string(),
        "hammerhead".to_string(),
        "electric ray".to_string(),
        "stingray".to_string(),
        "cock".to_string(),
        "hen".to_string(),
        "ostrich".to_string(),
        // ... would contain all 1000 ImageNet classes in real implementation
    ]
}

/// Estimate memory usage for inference
fn estimate_memory_usage(batch_size: usize, channels: usize, height: usize, width: usize) -> f32 {
    // Input tensor + intermediate activations + model parameters
    let input_size = batch_size * channels * height * width * 4; // 4 bytes per f32
    let model_size = 11_700_000 * 4; // ~11.7M parameters * 4 bytes
    let activation_size = input_size * 10; // Rough estimate for intermediate activations
    
    (input_size + model_size + activation_size) as f32 / (1024.0 * 1024.0)
}