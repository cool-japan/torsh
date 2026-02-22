use torsh::prelude::*;
use torsh::vision::{
    models::{create_model, ModelConfig},
    transforms::{Compose, Resize, CenterCrop, ToTensor, Normalize},
};

fn main() -> Result<()> {
    println!("ToRSh Vision Demo");
    println!("=================");
    
    // Create a ResNet-18 model
    let config = ModelConfig {
        num_classes: 1000,
        dropout: 0.2,
        pretrained: false,
    };
    
    let model = create_model("resnet18", config)?;
    println!("Created ResNet-18 model with 1000 classes");
    
    // List available models
    let models = torsh::vision::models::list_available_models();
    println!("\nAvailable models:");
    for (i, model_name) in models.iter().enumerate() {
        if i % 4 == 0 {
            print!("\n  ");
        }
        print!("{:<20}", model_name);
    }
    println!("\n");
    
    // Create a transform pipeline
    let transform = Compose::new(vec![
        Box::new(Resize::new((256, 256))),
        Box::new(CenterCrop::new((224, 224))),
    ]);
    
    // Create a dummy input tensor
    let input = randn(&[1, 3, 224, 224], DType::F32, Device::Cpu);
    println!("Input shape: {:?}", input.shape());
    
    // Forward pass
    let output = model.borrow().forward(&input);
    println!("Output shape: {:?}", output.shape());
    
    // Demonstrate image normalization
    let normalize = Normalize::new(
        vec![0.485, 0.456, 0.406],  // ImageNet mean
        vec![0.229, 0.224, 0.225],  // ImageNet std
    );
    
    println!("\nCreated ImageNet normalization transform");
    
    // Create different model architectures
    println!("\nCreating different architectures:");
    
    let vgg16 = create_model("vgg16", config)?;
    println!("- VGG-16 created");
    
    let mobilenet = create_model("mobilenet_v2_1.0", config)?;
    println!("- MobileNetV2 created");
    
    let efficientnet = create_model("efficientnet_b0", config)?;
    println!("- EfficientNet-B0 created");
    
    let vit = create_model("vit_tiny_patch16_224", config)?;
    println!("- Vision Transformer (ViT-Tiny) created");
    
    println!("\nVision module successfully demonstrated!");
    
    Ok(())
}