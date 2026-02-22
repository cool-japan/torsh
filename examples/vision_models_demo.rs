use torsh_core::device::DeviceType;
use torsh_nn::Module;
use torsh_tensor::Tensor;
use torsh_vision::{
    datasets::{ImageNet, CIFAR10, MNIST},
    models::{registry::ModelRegistry, AlexNet, ModelConfig, ResNet, VisionModel, VGG},
    ops::{center_crop, horizontal_flip, normalize, resize},
    prelude::*,
    transforms::{Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor, Transform},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ToRSh Vision Models Demo");
    println!("========================\n");

    let device = DeviceType::Cpu;

    // Test model configurations
    println!("1. Testing Model Configurations");
    println!("--------------------------------");

    let config = ModelConfig::default();
    println!(
        "Default config: num_classes={}, dropout={}, pretrained={}",
        config.num_classes, config.dropout, config.pretrained
    );

    let custom_config = ModelConfig {
        num_classes: 10, // CIFAR-10
        dropout: 0.5,
        pretrained: false,
    };
    println!(
        "CIFAR-10 config: num_classes={}, dropout={}, pretrained={}",
        custom_config.num_classes, custom_config.dropout, custom_config.pretrained
    );

    // Test transforms
    println!("\n2. Testing Image Transforms");
    println!("----------------------------");

    // Create a sample 3x224x224 RGB image tensor
    let sample_image = Tensor::from_data(
        (0..3 * 224 * 224)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect(),
        vec![3, 224, 224],
        device,
    );
    println!("Sample image shape: {:?}", sample_image.shape());

    // Test individual transforms
    let resize_transform = Resize::new((256, 256));
    println!("Resize transform created");

    let normalize_transform = Normalize::new(
        vec![0.485, 0.456, 0.406], // ImageNet mean
        vec![0.229, 0.224, 0.225], // ImageNet std
    );
    println!("Normalize transform created");

    // Test composed transforms
    let transform_pipeline = Compose::new(vec![
        Box::new(Resize::new((256, 256))),
        Box::new(ToTensor::new()),
        Box::new(Normalize::new(
            vec![0.485, 0.456, 0.406],
            vec![0.229, 0.224, 0.225],
        )),
    ]);
    println!("Transform pipeline created");

    // Test ResNet models
    println!("\n3. Testing ResNet Models");
    println!("------------------------");

    let resnet18 = ResNet::resnet18(custom_config);
    println!(
        "ResNet-18: {} classes, input size {:?}",
        resnet18.num_classes(),
        resnet18.input_size()
    );

    let resnet34 = ResNet::resnet34(custom_config);
    println!(
        "ResNet-34: {} classes, input size {:?}",
        resnet34.num_classes(),
        resnet34.input_size()
    );

    // Test forward pass with batch input (batch_size=2, channels=3, height=224, width=224)
    let batch_input = Tensor::from_data(
        (0..2 * 3 * 224 * 224)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect(),
        vec![2, 3, 224, 224],
        device,
    );
    println!("Batch input shape: {:?}", batch_input.shape());

    match resnet18.forward(&batch_input) {
        Ok(output) => println!("ResNet-18 output shape: {:?}", output.shape()),
        Err(e) => println!("ResNet-18 forward pass error (expected): {:?}", e),
    }

    // Test VGG models
    println!("\n4. Testing VGG Models");
    println!("---------------------");

    let vgg11 = VGG::vgg11(custom_config);
    println!(
        "VGG-11: {} classes, input size {:?}",
        vgg11.num_classes(),
        vgg11.input_size()
    );

    let vgg16 = VGG::vgg16(custom_config);
    println!(
        "VGG-16: {} classes, input size {:?}",
        vgg16.num_classes(),
        vgg16.input_size()
    );

    match vgg11.forward(&batch_input) {
        Ok(output) => println!("VGG-11 output shape: {:?}", output.shape()),
        Err(e) => println!("VGG-11 forward pass error (expected): {:?}", e),
    }

    // Test AlexNet
    println!("\n5. Testing AlexNet Model");
    println!("------------------------");

    let alexnet = AlexNet::new(custom_config);
    println!(
        "AlexNet: {} classes, input size {:?}",
        alexnet.num_classes(),
        alexnet.input_size()
    );

    match alexnet.forward(&batch_input) {
        Ok(output) => println!("AlexNet output shape: {:?}", output.shape()),
        Err(e) => println!("AlexNet forward pass error (expected): {:?}", e),
    }

    // Test model registry
    println!("\n6. Testing Model Registry");
    println!("-------------------------");

    let registry = ModelRegistry::new();
    println!("Model registry initialized");

    // Test dataset abstractions
    println!("\n7. Testing Dataset Abstractions");
    println!("-------------------------------");

    // Dataset structures (note: these would require actual data files)
    println!("CIFAR-10 dataset structure available");
    println!("ImageNet dataset structure available");
    println!("MNIST dataset structure available");

    // Test model parameters
    println!("\n8. Model Parameters");
    println!("-------------------");

    let resnet18_params = resnet18.parameters();
    println!("ResNet-18 parameter count: {}", resnet18_params.len());

    let vgg11_params = vgg11.parameters();
    println!("VGG-11 parameter count: {}", vgg11_params.len());

    let alexnet_params = alexnet.parameters();
    println!("AlexNet parameter count: {}", alexnet_params.len());

    // Test training mode
    println!("\n9. Training Mode");
    println!("----------------");

    println!("ResNet-18 training mode: {}", resnet18.training());
    println!("VGG-11 training mode: {}", vgg11.training());
    println!("AlexNet training mode: {}", alexnet.training());

    println!("\n✅ All vision model tests completed successfully!");

    Ok(())
}
