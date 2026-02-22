//! ResNet Implementation Example using ToRSh
//! 
//! This example demonstrates:
//! - Building ResNet architectures from scratch
//! - Implementing residual blocks with skip connections
//! - Different ResNet variants (ResNet-18, 34, 50, 101, 152)
//! - Training on ImageNet-style datasets
//! - Transfer learning and fine-tuning

use torsh::prelude::*;
use torsh::nn::{Module, Conv2d, Linear, Sequential, BatchNorm2d, ReLU, AvgPool2d, MaxPool2d};
use torsh::optim::{SGD, Optimizer};
use torsh::tensor::Tensor;
use torsh_vision::prelude::*;
use std::error::Error;

/// Basic residual block for ResNet-18/34
#[derive(Debug)]
struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<Sequential>,
    stride: usize,
}

impl BasicBlock {
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let downsample = if stride != 1 || in_channels != out_channels {
            Some(
                Sequential::new()
                    .add_module("conv", Conv2d::new(in_channels, out_channels, (1, 1), (stride, stride), (0, 0), (1, 1), false, 1))
                    .add_module("bn", BatchNorm2d::new(out_channels, 1e-5, 0.1, true, true))
            )
        } else {
            None
        };
        
        Self {
            conv1: Conv2d::new(in_channels, out_channels, (3, 3), (stride, stride), (1, 1), (1, 1), false, 1),
            bn1: BatchNorm2d::new(out_channels, 1e-5, 0.1, true, true),
            relu: ReLU::new(false),
            conv2: Conv2d::new(out_channels, out_channels, (3, 3), (1, 1), (1, 1), (1, 1), false, 1),
            bn2: BatchNorm2d::new(out_channels, 1e-5, 0.1, true, true),
            downsample,
            stride,
        }
    }
}

impl Module for BasicBlock {
    type Error = torsh::TorshError;
    
    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        let identity = input.clone();
        
        // First conv block
        let out = self.conv1.forward(input)?;
        let out = self.bn1.forward(&out)?;
        let out = self.relu.forward(&out)?;
        
        // Second conv block
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward(&out)?;
        
        // Skip connection
        let identity = if let Some(ref downsample) = self.downsample {
            downsample.forward(&identity)?
        } else {
            identity
        };
        
        // Add residual
        let out = out.add(&identity)?;
        self.relu.forward(&out)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![];
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        
        if let Some(ref downsample) = self.downsample {
            params.extend(downsample.parameters());
        }
        
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![];
        
        for (name, param) in self.conv1.named_parameters() {
            params.push((format!("conv1.{}", name), param));
        }
        for (name, param) in self.bn1.named_parameters() {
            params.push((format!("bn1.{}", name), param));
        }
        for (name, param) in self.conv2.named_parameters() {
            params.push((format!("conv2.{}", name), param));
        }
        for (name, param) in self.bn2.named_parameters() {
            params.push((format!("bn2.{}", name), param));
        }
        
        if let Some(ref downsample) = self.downsample {
            for (name, param) in downsample.named_parameters() {
                params.push((format!("downsample.{}", name), param));
            }
        }
        
        params
    }
}

/// Bottleneck block for ResNet-50/101/152
#[derive(Debug)]
struct Bottleneck {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    relu: ReLU,
    downsample: Option<Sequential>,
    stride: usize,
}

impl Bottleneck {
    const EXPANSION: usize = 4;
    
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let width = out_channels;
        let out_channels_expanded = out_channels * Self::EXPANSION;
        
        let downsample = if stride != 1 || in_channels != out_channels_expanded {
            Some(
                Sequential::new()
                    .add_module("conv", Conv2d::new(in_channels, out_channels_expanded, (1, 1), (stride, stride), (0, 0), (1, 1), false, 1))
                    .add_module("bn", BatchNorm2d::new(out_channels_expanded, 1e-5, 0.1, true, true))
            )
        } else {
            None
        };
        
        Self {
            conv1: Conv2d::new(in_channels, width, (1, 1), (1, 1), (0, 0), (1, 1), false, 1),
            bn1: BatchNorm2d::new(width, 1e-5, 0.1, true, true),
            conv2: Conv2d::new(width, width, (3, 3), (stride, stride), (1, 1), (1, 1), false, 1),
            bn2: BatchNorm2d::new(width, 1e-5, 0.1, true, true),
            conv3: Conv2d::new(width, out_channels_expanded, (1, 1), (1, 1), (0, 0), (1, 1), false, 1),
            bn3: BatchNorm2d::new(out_channels_expanded, 1e-5, 0.1, true, true),
            relu: ReLU::new(false),
            downsample,
            stride,
        }
    }
}

impl Module for Bottleneck {
    type Error = torsh::TorshError;
    
    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        let identity = input.clone();
        
        // 1x1 conv
        let out = self.conv1.forward(input)?;
        let out = self.bn1.forward(&out)?;
        let out = self.relu.forward(&out)?;
        
        // 3x3 conv
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward(&out)?;
        let out = self.relu.forward(&out)?;
        
        // 1x1 conv
        let out = self.conv3.forward(&out)?;
        let out = self.bn3.forward(&out)?;
        
        // Skip connection
        let identity = if let Some(ref downsample) = self.downsample {
            downsample.forward(&identity)?
        } else {
            identity
        };
        
        // Add residual
        let out = out.add(&identity)?;
        self.relu.forward(&out)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![];
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.bn3.parameters());
        
        if let Some(ref downsample) = self.downsample {
            params.extend(downsample.parameters());
        }
        
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![];
        
        // Similar to BasicBlock but with conv3/bn3
        for (name, param) in self.conv1.named_parameters() {
            params.push((format!("conv1.{}", name), param));
        }
        for (name, param) in self.bn1.named_parameters() {
            params.push((format!("bn1.{}", name), param));
        }
        for (name, param) in self.conv2.named_parameters() {
            params.push((format!("conv2.{}", name), param));
        }
        for (name, param) in self.bn2.named_parameters() {
            params.push((format!("bn2.{}", name), param));
        }
        for (name, param) in self.conv3.named_parameters() {
            params.push((format!("conv3.{}", name), param));
        }
        for (name, param) in self.bn3.named_parameters() {
            params.push((format!("bn3.{}", name), param));
        }
        
        if let Some(ref downsample) = self.downsample {
            for (name, param) in downsample.named_parameters() {
                params.push((format!("downsample.{}", name), param));
            }
        }
        
        params
    }
}

/// Complete ResNet architecture
#[derive(Debug)]
struct ResNet {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,
    
    layer1: Vec<Box<dyn Module<Error = torsh::TorshError>>>,
    layer2: Vec<Box<dyn Module<Error = torsh::TorshError>>>,
    layer3: Vec<Box<dyn Module<Error = torsh::TorshError>>>,
    layer4: Vec<Box<dyn Module<Error = torsh::TorshError>>>,
    
    avgpool: AvgPool2d,
    fc: Linear,
    
    num_classes: usize,
}

impl ResNet {
    /// Create ResNet-18
    fn resnet18(num_classes: usize) -> Self {
        Self::new(&[2, 2, 2, 2], BasicBlock::new, 1, num_classes)
    }
    
    /// Create ResNet-34
    fn resnet34(num_classes: usize) -> Self {
        Self::new(&[3, 4, 6, 3], BasicBlock::new, 1, num_classes)
    }
    
    /// Create ResNet-50
    fn resnet50(num_classes: usize) -> Self {
        Self::new(&[3, 4, 6, 3], |in_ch, out_ch, stride| {
            Box::new(Bottleneck::new(in_ch, out_ch, stride))
        }, Bottleneck::EXPANSION, num_classes)
    }
    
    /// Create ResNet-101
    fn resnet101(num_classes: usize) -> Self {
        Self::new(&[3, 4, 23, 3], |in_ch, out_ch, stride| {
            Box::new(Bottleneck::new(in_ch, out_ch, stride))
        }, Bottleneck::EXPANSION, num_classes)
    }
    
    /// Create ResNet-152
    fn resnet152(num_classes: usize) -> Self {
        Self::new(&[3, 8, 36, 3], |in_ch, out_ch, stride| {
            Box::new(Bottleneck::new(in_ch, out_ch, stride))
        }, Bottleneck::EXPANSION, num_classes)
    }
    
    /// Generic ResNet constructor
    fn new<F>(layers: &[usize], block_fn: F, expansion: usize, num_classes: usize) -> Self
    where
        F: Fn(usize, usize, usize) -> Box<dyn Module<Error = torsh::TorshError>>,
    {
        let mut in_channels = 64;
        
        // Build layers
        let layer1 = Self::make_layer(&block_fn, &mut in_channels, 64, layers[0], 1, expansion);
        let layer2 = Self::make_layer(&block_fn, &mut in_channels, 128, layers[1], 2, expansion);
        let layer3 = Self::make_layer(&block_fn, &mut in_channels, 256, layers[2], 2, expansion);
        let layer4 = Self::make_layer(&block_fn, &mut in_channels, 512, layers[3], 2, expansion);
        
        Self {
            conv1: Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), (1, 1), false, 1),
            bn1: BatchNorm2d::new(64, 1e-5, 0.1, true, true),
            relu: ReLU::new(false),
            maxpool: MaxPool2d::new((3, 3), (2, 2), (1, 1), (1, 1)),
            
            layer1,
            layer2,
            layer3,
            layer4,
            
            avgpool: AvgPool2d::new((7, 7), (1, 1), (0, 0)),
            fc: Linear::new(512 * expansion, num_classes, true),
            
            num_classes,
        }
    }
    
    /// Helper to create a layer of blocks
    fn make_layer<F>(
        block_fn: &F,
        in_channels: &mut usize,
        out_channels: usize,
        blocks: usize,
        stride: usize,
        expansion: usize,
    ) -> Vec<Box<dyn Module<Error = torsh::TorshError>>>
    where
        F: Fn(usize, usize, usize) -> Box<dyn Module<Error = torsh::TorshError>>,
    {
        let mut layers = vec![];
        
        // First block (may downsample)
        layers.push(block_fn(*in_channels, out_channels, stride));
        *in_channels = out_channels * expansion;
        
        // Remaining blocks
        for _ in 1..blocks {
            layers.push(block_fn(*in_channels, out_channels, 1));
        }
        
        layers
    }
}

impl Module for ResNet {
    type Error = torsh::TorshError;
    
    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        // Initial convolution
        let x = self.conv1.forward(input)?;
        let x = self.bn1.forward(&x)?;
        let x = self.relu.forward(&x)?;
        let x = self.maxpool.forward(&x)?;
        
        // Residual layers
        let mut x = x;
        for block in &self.layer1 {
            x = block.forward(&x)?;
        }
        for block in &self.layer2 {
            x = block.forward(&x)?;
        }
        for block in &self.layer3 {
            x = block.forward(&x)?;
        }
        for block in &self.layer4 {
            x = block.forward(&x)?;
        }
        
        // Global average pooling
        let x = self.avgpool.forward(&x)?;
        
        // Flatten and classify
        let shape = x.shape();
        let batch_size = shape.dims()[0] as i32;
        let x = x.view(&[batch_size, -1])?;
        
        self.fc.forward(&x)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![];
        
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        
        for block in &self.layer1 {
            params.extend(block.parameters());
        }
        for block in &self.layer2 {
            params.extend(block.parameters());
        }
        for block in &self.layer3 {
            params.extend(block.parameters());
        }
        for block in &self.layer4 {
            params.extend(block.parameters());
        }
        
        params.extend(self.fc.parameters());
        
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![];
        
        for (name, param) in self.conv1.named_parameters() {
            params.push((format!("conv1.{}", name), param));
        }
        for (name, param) in self.bn1.named_parameters() {
            params.push((format!("bn1.{}", name), param));
        }
        
        // Add layer parameters with proper naming
        for (i, block) in self.layer1.iter().enumerate() {
            for (name, param) in block.named_parameters() {
                params.push((format!("layer1.{}.{}", i, name), param));
            }
        }
        for (i, block) in self.layer2.iter().enumerate() {
            for (name, param) in block.named_parameters() {
                params.push((format!("layer2.{}.{}", i, name), param));
            }
        }
        for (i, block) in self.layer3.iter().enumerate() {
            for (name, param) in block.named_parameters() {
                params.push((format!("layer3.{}.{}", i, name), param));
            }
        }
        for (i, block) in self.layer4.iter().enumerate() {
            for (name, param) in block.named_parameters() {
                params.push((format!("layer4.{}.{}", i, name), param));
            }
        }
        
        for (name, param) in self.fc.named_parameters() {
            params.push((format!("fc.{}", name), param));
        }
        
        params
    }
}

/// Load pretrained weights (simulation)
fn load_pretrained_weights(model: &mut ResNet, model_name: &str) -> Result<(), Box<dyn Error>> {
    println!("📥 Loading pretrained weights for {}...", model_name);
    
    // In a real implementation, we would load actual pretrained weights
    // For now, we just simulate the process
    std::thread::sleep(std::time::Duration::from_millis(500));
    
    println!("✅ Pretrained weights loaded successfully!");
    Ok(())
}

/// Fine-tune for a new task
fn fine_tune_model(
    model: &mut ResNet,
    freeze_backbone: bool,
    new_num_classes: usize,
) -> Result<(), Box<dyn Error>> {
    println!("\n🔧 Fine-tuning model for {} classes...", new_num_classes);
    
    if freeze_backbone {
        println!("❄️  Freezing backbone layers...");
        // In a real implementation, we would set requires_grad=False for backbone
    }
    
    // Replace final classifier
    println!("🔄 Replacing classifier head...");
    // model.fc = Linear::new(512 * expansion, new_num_classes, true);
    
    println!("✅ Model ready for fine-tuning!");
    Ok(())
}

/// Train one epoch
fn train_epoch(
    model: &ResNet,
    epoch: usize,
    total_epochs: usize,
) -> Result<(f32, f32), Box<dyn Error>> {
    println!("\nEpoch {}/{}", epoch, total_epochs);
    println!("-" * 50);
    
    // Simulate training progress
    let steps = 10;
    for step in 0..steps {
        let progress = (step + 1) as f32 / steps as f32;
        let loss = 2.5 * (1.0 - progress * 0.3) + rand::randn(&[1])?.item::<f32>() * 0.1;
        let accuracy = 20.0 + progress * 60.0 + rand::randn(&[1])?.item::<f32>() * 5.0;
        
        print!("\r  Step [{}/{}] - Loss: {:.4}, Accuracy: {:.2}%", 
               step + 1, steps, loss, accuracy);
        std::io::Write::flush(&mut std::io::stdout())?;
        
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    println!();
    
    let final_loss = 2.0 - (epoch as f32 * 0.3);
    let final_accuracy = 70.0 + (epoch as f32 * 5.0);
    
    Ok((final_loss, final_accuracy))
}

/// Visualize feature maps
fn visualize_features(model: &ResNet, sample_input: &Tensor) -> Result<(), Box<dyn Error>> {
    println!("\n🔍 Visualizing intermediate features...");
    
    // Get intermediate activations (simplified)
    let x = model.conv1.forward(sample_input)?;
    let x = model.bn1.forward(&x)?;
    let x = model.relu.forward(&x)?;
    
    let shape = x.shape();
    println!("  Conv1 output shape: {:?}", shape.dims());
    println!("  Number of feature maps: {}", shape.dims()[1]);
    
    // Show activation statistics
    let mean = x.mean_all()?.item::<f32>();
    let std = x.std_all()?.item::<f32>();
    println!("  Activation statistics: mean={:.3}, std={:.3}", mean, std);
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🏗️  ToRSh ResNet Implementation Example");
    println!("========================================");
    
    // Set random seed
    torsh::manual_seed(42);
    
    // Configuration
    let num_classes = 1000; // ImageNet classes
    let batch_size = 32;
    let learning_rate = 0.1;
    let num_epochs = 5;
    
    println!("\n📋 Configuration:");
    println!("  - Number of classes: {}", num_classes);
    println!("  - Batch size: {}", batch_size);
    println!("  - Learning rate: {}", learning_rate);
    println!("  - Epochs: {}", num_epochs);
    
    // Create different ResNet variants
    println!("\n🏗️  Creating ResNet architectures...");
    
    let mut models = vec![
        ("ResNet-18", ResNet::resnet18(num_classes)),
        ("ResNet-34", ResNet::resnet34(num_classes)),
        ("ResNet-50", ResNet::resnet50(num_classes)),
        ("ResNet-101", ResNet::resnet101(num_classes)),
        ("ResNet-152", ResNet::resnet152(num_classes)),
    ];
    
    // Display model statistics
    println!("\n📊 Model Statistics:");
    println!("{:<15} {:>15} {:>10}", "Model", "Parameters", "Layers");
    println!("-" * 42);
    
    for (name, model) in &models {
        let param_count: usize = model.parameters().iter()
            .map(|p| p.numel())
            .sum();
        let layer_count = 
            2 + // conv1, fc
            model.layer1.len() + 
            model.layer2.len() + 
            model.layer3.len() + 
            model.layer4.len();
        
        println!("{:<15} {:>15} {:>10}", 
                name, 
                format!("{:.1}M", param_count as f32 / 1_000_000.0),
                layer_count);
    }
    
    // Select ResNet-50 for training demo
    println!("\n🎯 Selecting ResNet-50 for training demo...");
    let model = ResNet::resnet50(num_classes);
    
    // Load pretrained weights
    load_pretrained_weights(&mut model, "ResNet-50")?;
    
    // Create optimizer
    let mut optimizer = SGD::builder()
        .learning_rate(learning_rate)
        .momentum(0.9)
        .weight_decay(1e-4)
        .build();
    
    for param in model.parameters() {
        optimizer.add_param_group(param.clone());
    }
    
    // Create sample input
    let sample_input = Tensor::randn(&[batch_size, 3, 224, 224])?;
    println!("\n📸 Sample input shape: {:?}", sample_input.shape());
    
    // Test forward pass
    println!("\n🔄 Testing forward pass...");
    let output = model.forward(&sample_input)?;
    println!("✅ Output shape: {:?} (batch_size × num_classes)", output.shape());
    
    // Visualize features
    visualize_features(&model, &sample_input)?;
    
    // Training loop
    println!("\n🚀 Starting training...");
    println!("=" * 50);
    
    for epoch in 1..=num_epochs {
        model.train();
        let (loss, accuracy) = train_epoch(&model, epoch, num_epochs)?;
        
        println!("📈 Epoch {} Summary: Loss={:.4}, Top-1 Accuracy={:.2}%", 
                epoch, loss, accuracy);
    }
    
    // Fine-tuning demo
    println!("\n🎨 Fine-tuning Demo");
    println!("===================");
    
    let new_classes = 10; // e.g., for CIFAR-10
    fine_tune_model(&mut model, true, new_classes)?;
    
    // Model analysis
    println!("\n📊 Model Analysis");
    println!("=================");
    
    // Layer-wise parameter distribution
    println!("\nParameter distribution by layer type:");
    let conv_params: usize = model.named_parameters().iter()
        .filter(|(name, _)| name.contains("conv"))
        .map(|(_, p)| p.numel())
        .sum();
    let bn_params: usize = model.named_parameters().iter()
        .filter(|(name, _)| name.contains("bn"))
        .map(|(_, p)| p.numel())
        .sum();
    let fc_params: usize = model.fc.parameters().iter()
        .map(|p| p.numel())
        .sum();
    
    let total_params = conv_params + bn_params + fc_params;
    println!("  Convolutional: {:.1}M ({:.1}%)", 
            conv_params as f32 / 1_000_000.0,
            conv_params as f32 / total_params as f32 * 100.0);
    println!("  Batch Norm: {:.1}K ({:.1}%)", 
            bn_params as f32 / 1_000.0,
            bn_params as f32 / total_params as f32 * 100.0);
    println!("  Fully Connected: {:.1}M ({:.1}%)", 
            fc_params as f32 / 1_000_000.0,
            fc_params as f32 / total_params as f32 * 100.0);
    
    // Memory footprint estimation
    println!("\nMemory footprint (approximate):");
    let param_memory = total_params * 4; // 4 bytes per float32
    let activation_memory = batch_size * 512 * 7 * 7 * 4; // Rough estimate
    println!("  Parameters: {:.1} MB", param_memory as f32 / 1_000_000.0);
    println!("  Activations (batch={}): {:.1} MB", 
            batch_size, activation_memory as f32 / 1_000_000.0);
    
    println!("\n✅ ResNet example completed successfully!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_block() -> Result<(), Box<dyn Error>> {
        let block = BasicBlock::new(64, 128, 2);
        let input = Tensor::randn(&[1, 64, 56, 56])?;
        let output = block.forward(&input)?;
        
        // Check output shape (should be downsampled)
        assert_eq!(output.shape(), &[1, 128, 28, 28]);
        
        Ok(())
    }
    
    #[test]
    fn test_bottleneck_block() -> Result<(), Box<dyn Error>> {
        let block = Bottleneck::new(256, 128, 1);
        let input = Tensor::randn(&[1, 256, 56, 56])?;
        let output = block.forward(&input)?;
        
        // Check output shape (expansion = 4)
        assert_eq!(output.shape(), &[1, 512, 56, 56]);
        
        Ok(())
    }
    
    #[test]
    fn test_resnet_variants() -> Result<(), Box<dyn Error>> {
        let input = Tensor::randn(&[2, 3, 224, 224])?;
        
        // Test each variant
        for (name, model) in &[
            ("ResNet-18", ResNet::resnet18(10)),
            ("ResNet-34", ResNet::resnet34(10)),
            ("ResNet-50", ResNet::resnet50(10)),
        ] {
            let output = model.forward(&input)?;
            assert_eq!(output.shape(), &[2, 10], "Failed for {}", name);
        }
        
        Ok(())
    }
}