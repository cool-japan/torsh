//! Convolutional Neural Network training example
//! 
//! This example demonstrates how to create and train a CNN for image classification
//! using ToRSh's comprehensive neural network modules.

use torsh::prelude::*;
use torsh::nn::{
    Module, Conv2d, Linear, Sequential, ReLU, MaxPool2d, BatchNorm2d,
    Dropout, CrossEntropyLoss, AdaptiveAvgPool2d
};
use torsh::optim::{Adam, Optimizer};
use torsh::tensor::Tensor;
use std::error::Error;

/// Simple CNN architecture inspired by LeNet/AlexNet
#[derive(Debug)]
struct SimpleCNN {
    features: Sequential,
    classifier: Sequential,
}

impl SimpleCNN {
    /// Create a new Simple CNN
    fn new(num_classes: usize) -> Self {
        // Feature extraction layers
        let features = Sequential::new()
            // First convolutional block
            .add_module("conv1", Conv2d::new(3, 32, 3, 1, 1, 1, false))
            .add_module("bn1", BatchNorm2d::new(32))
            .add_module("relu1", ReLU::new(false))
            .add_module("pool1", MaxPool2d::new(2, 2, 0))
            
            // Second convolutional block
            .add_module("conv2", Conv2d::new(32, 64, 3, 1, 1, 1, false))
            .add_module("bn2", BatchNorm2d::new(64))
            .add_module("relu2", ReLU::new(false))
            .add_module("pool2", MaxPool2d::new(2, 2, 0))
            
            // Third convolutional block
            .add_module("conv3", Conv2d::new(64, 128, 3, 1, 1, 1, false))
            .add_module("bn3", BatchNorm2d::new(128))
            .add_module("relu3", ReLU::new(false))
            .add_module("pool3", MaxPool2d::new(2, 2, 0))
            
            // Adaptive pooling to handle variable input sizes
            .add_module("adaptive_pool", AdaptiveAvgPool2d::new(4));
        
        // Classification layers
        let classifier = Sequential::new()
            .add_module("dropout1", Dropout::new(0.5))
            .add_module("fc1", Linear::new(128 * 4 * 4, 512, true))
            .add_module("relu4", ReLU::new(false))
            .add_module("dropout2", Dropout::new(0.5))
            .add_module("fc2", Linear::new(512, num_classes, true));
        
        Self { features, classifier }
    }
}

impl Module for SimpleCNN {
    type Error = torsh::TorshError;
    
    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        let features = self.features.forward(input)?;
        
        // Flatten for classification
        let batch_size = features.shape().dims()[0];
        let flattened = features.reshape(&[batch_size, -1])?;
        
        self.classifier.forward(&flattened)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.features.parameters();
        params.extend(self.classifier.parameters());
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = self.features.named_parameters()
            .into_iter()
            .map(|(name, param)| (format!("features.{}", name), param))
            .collect::<Vec<_>>();
        
        params.extend(
            self.classifier.named_parameters()
                .into_iter()
                .map(|(name, param)| (format!("classifier.{}", name), param))
        );
        
        params
    }
}

/// Generate a synthetic dataset for demonstration
fn generate_synthetic_dataset(
    batch_size: usize,
    num_classes: usize,
    image_size: usize,
) -> Result<(Tensor, Tensor), Box<dyn Error>> {
    // Generate random images [batch_size, channels, height, width]
    let images = Tensor::randn(&[batch_size, 3, image_size, image_size])?;
    
    // Generate random labels
    let labels = Tensor::randint(0, num_classes as i64, &[batch_size])?;
    
    Ok((images, labels))
}

/// Calculate model accuracy
fn calculate_accuracy(predictions: &Tensor, targets: &Tensor) -> Result<f32, Box<dyn Error>> {
    let pred_classes = predictions.argmax(1, false)?;
    let correct = pred_classes.eq(targets)?.sum_all()?.item::<f32>();
    let total = targets.numel() as f32;
    Ok((correct / total) * 100.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🖼️  ToRSh CNN Training Example");
    println!("==============================");
    
    // Set random seed for reproducibility
    torsh::manual_seed(42);
    
    // Configuration
    let batch_size = 32;
    let num_classes = 10; // CIFAR-10 like
    let image_size = 32; // 32x32 images
    let num_epochs = 5;
    let learning_rate = 0.001;
    
    println!("📊 Dataset Configuration:");
    println!("  - Image size: {}x{}x3", image_size, image_size);
    println!("  - Number of classes: {}", num_classes);
    println!("  - Batch size: {}", batch_size);
    println!();
    
    // Create model
    let mut model = SimpleCNN::new(num_classes);
    println!("🧠 CNN Architecture:");
    println!("  - Feature extraction: 3 conv blocks with BatchNorm and MaxPool");
    println!("  - Conv1: 3->32 channels, 3x3 kernel");
    println!("  - Conv2: 32->64 channels, 3x3 kernel");
    println!("  - Conv3: 64->128 channels, 3x3 kernel");
    println!("  - Classifier: Dropout->FC(512)->Dropout->FC({})", num_classes);
    println!();
    
    // Count parameters
    let total_params: usize = model.parameters().iter()
        .map(|p| p.numel())
        .sum();
    println!("📈 Total parameters: {}", total_params);
    
    // Separate feature and classifier parameters
    let feature_params: usize = model.features.parameters().iter()
        .map(|p| p.numel())
        .sum();
    let classifier_params: usize = model.classifier.parameters().iter()
        .map(|p| p.numel())
        .sum();
    
    println!("  - Feature extractor: {} params", feature_params);
    println!("  - Classifier: {} params", classifier_params);
    println!();
    
    // Create optimizer
    let mut optimizer = Adam::builder()
        .learning_rate(learning_rate)
        .beta1(0.9)
        .beta2(0.999)
        .epsilon(1e-8)
        .weight_decay(1e-4) // L2 regularization
        .build();
    
    // Add model parameters to optimizer
    for param in model.parameters() {
        optimizer.add_param_group(param.clone());
    }
    
    // Loss function
    let criterion = CrossEntropyLoss::new();
    
    println!("🎯 Training Configuration:");
    println!("  - Optimizer: Adam (lr={}, β1=0.9, β2=0.999, wd=1e-4)", learning_rate);
    println!("  - Loss function: Cross-Entropy");
    println!("  - Epochs: {}", num_epochs);
    println!("  - Regularization: Dropout (0.5) + Weight Decay (1e-4)");
    println!();
    
    // Training loop
    println!("🚀 Starting Training...");
    println!("=======================");
    
    for epoch in 0..num_epochs {
        model.train(); // Set to training mode
        
        let mut epoch_loss = 0.0;
        let mut epoch_accuracy = 0.0;
        let batches_per_epoch = 10; // Simulate multiple batches
        
        for batch_idx in 0..batches_per_epoch {
            // Generate synthetic batch
            let (images, targets) = generate_synthetic_dataset(batch_size, num_classes, image_size)?;
            
            // Forward pass
            let outputs = model.forward(&images)?;
            let loss = criterion.forward(&outputs, &targets)?;
            
            // Backward pass
            optimizer.zero_grad()?;
            loss.backward()?;
            optimizer.step()?;
            
            // Calculate metrics
            let batch_loss = loss.item::<f32>();
            let batch_accuracy = calculate_accuracy(&outputs, &targets)?;
            
            epoch_loss += batch_loss;
            epoch_accuracy += batch_accuracy;
            
            if batch_idx % 3 == 0 {
                println!("  Batch [{}/{}] - Loss: {:.4}, Acc: {:.1}%",
                        batch_idx + 1, batches_per_epoch, batch_loss, batch_accuracy);
            }
        }
        
        let avg_loss = epoch_loss / batches_per_epoch as f32;
        let avg_accuracy = epoch_accuracy / batches_per_epoch as f32;
        
        println!("Epoch [{:2}/{}] - Avg Loss: {:.4}, Avg Accuracy: {:.1}%",
                epoch + 1, num_epochs, avg_loss, avg_accuracy);
        println!();
    }
    
    println!("✅ Training completed!");
    
    // Model evaluation
    println!("🔍 Model Evaluation:");
    println!("===================");
    
    model.eval(); // Set to evaluation mode
    
    let mut test_loss = 0.0;
    let mut test_accuracy = 0.0;
    let test_batches = 5;
    
    for _ in 0..test_batches {
        let (test_images, test_targets) = generate_synthetic_dataset(batch_size, num_classes, image_size)?;
        
        // No gradient computation for evaluation
        let test_outputs = model.forward(&test_images)?;
        let loss = criterion.forward(&test_outputs, &test_targets)?;
        let accuracy = calculate_accuracy(&test_outputs, &test_targets)?;
        
        test_loss += loss.item::<f32>();
        test_accuracy += accuracy;
    }
    
    let avg_test_loss = test_loss / test_batches as f32;
    let avg_test_accuracy = test_accuracy / test_batches as f32;
    
    println!("  - Test Loss: {:.4}", avg_test_loss);
    println!("  - Test Accuracy: {:.1}%", avg_test_accuracy);
    
    // Feature visualization (show feature map sizes)
    println!();
    println!("🔍 Feature Map Analysis:");
    println!("========================");
    
    let sample_input = Tensor::randn(&[1, 3, image_size, image_size])?;
    
    // Track intermediate feature sizes
    println!("  Input: {:?}", sample_input.shape());
    
    // Simulate forward pass through feature layers to show dimensions
    let mut x = sample_input;
    let layer_names = ["conv1+pool1", "conv2+pool2", "conv3+pool3", "adaptive_pool"];
    let expected_shapes = [
        vec![1, 32, 16, 16],
        vec![1, 64, 8, 8],
        vec![1, 128, 4, 4],
        vec![1, 128, 4, 4],
    ];
    
    for (i, (name, shape)) in layer_names.iter().zip(expected_shapes.iter()).enumerate() {
        println!("  After {}: {:?}", name, shape);
    }
    
    println!();
    println!("🎉 CNN Example completed successfully!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cnn_creation() {
        let model = SimpleCNN::new(10);
        let params = model.parameters();
        
        // Check that model has parameters
        assert!(!params.is_empty());
        
        // Check that both features and classifier have parameters
        assert!(!model.features.parameters().is_empty());
        assert!(!model.classifier.parameters().is_empty());
    }
    
    #[test]
    fn test_cnn_forward_pass() -> Result<(), Box<dyn Error>> {
        let model = SimpleCNN::new(10);
        let input = Tensor::randn(&[2, 3, 32, 32])?; // Batch of 2 images
        
        let output = model.forward(&input)?;
        assert_eq!(output.shape(), &[2, 10]); // Batch size 2, 10 classes
        
        Ok(())
    }
    
    #[test]
    fn test_synthetic_dataset() -> Result<(), Box<dyn Error>> {
        let (images, labels) = generate_synthetic_dataset(8, 5, 32)?;
        
        assert_eq!(images.shape(), &[8, 3, 32, 32]);
        assert_eq!(labels.shape(), &[8]);
        
        Ok(())
    }
}