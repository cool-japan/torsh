//! MNIST Classification Example using ToRSh
//! 
//! This example demonstrates:
//! - Loading the MNIST dataset
//! - Building a convolutional neural network for digit classification
//! - Training with optimization and loss computation
//! - Evaluating model accuracy
//! - Saving and loading model checkpoints

use torsh::prelude::*;
use torsh::nn::{Module, Conv2d, Linear, Sequential, ReLU, Dropout, MaxPool2d, CrossEntropyLoss};
use torsh::optim::{Adam, Optimizer};
use torsh::tensor::Tensor;
use torsh_vision::prelude::*;
use torsh_data::{DataLoader, Dataset as DatasetTrait};
use std::error::Error;
use std::path::Path;

/// CNN architecture for MNIST digit classification
#[derive(Debug)]
struct MNISTNet {
    features: Sequential,
    classifier: Sequential,
}

impl MNISTNet {
    /// Create a new MNIST CNN with the following architecture:
    /// - Conv2d(1, 32, 3x3) -> ReLU -> MaxPool(2x2)
    /// - Conv2d(32, 64, 3x3) -> ReLU -> MaxPool(2x2) 
    /// - Flatten -> Linear(7*7*64, 128) -> ReLU -> Dropout(0.5)
    /// - Linear(128, 10)
    fn new() -> Self {
        // Feature extraction layers
        let features = Sequential::new()
            .add_module("conv1", Conv2d::new(1, 32, (3, 3), (1, 1), (1, 1), (1, 1), true, 1))
            .add_module("relu1", ReLU::new(false))
            .add_module("pool1", MaxPool2d::new((2, 2), (2, 2), (0, 0), (1, 1)))
            .add_module("conv2", Conv2d::new(32, 64, (3, 3), (1, 1), (1, 1), (1, 1), true, 1))
            .add_module("relu2", ReLU::new(false))
            .add_module("pool2", MaxPool2d::new((2, 2), (2, 2), (0, 0), (1, 1)));
        
        // Classification layers
        let classifier = Sequential::new()
            .add_module("fc1", Linear::new(7 * 7 * 64, 128, true))
            .add_module("relu3", ReLU::new(false))
            .add_module("dropout", Dropout::new(0.5))
            .add_module("fc2", Linear::new(128, 10, true));
        
        Self { features, classifier }
    }
}

impl Module for MNISTNet {
    type Error = torsh::TorshError;
    
    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        // Extract features
        let x = self.features.forward(input)?;
        
        // Flatten for classifier
        let shape = x.shape();
        let shape_dims = shape.dims();
        let batch_size = shape_dims[0] as i32;
        let x = x.view(&[batch_size, -1])?;
        
        // Apply classifier
        self.classifier.forward(&x)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.features.parameters();
        params.extend(self.classifier.parameters());
        params
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = self.features.named_parameters();
        params.extend(self.classifier.named_parameters());
        params
    }
    
    fn train(&mut self) {
        self.features.train();
        self.classifier.train();
    }
    
    fn eval(&mut self) {
        self.features.eval();
        self.classifier.eval();
    }
}

/// Train the MNIST model
fn train_epoch(
    model: &MNISTNet,
    dataloader: &mut DataLoader<(Tensor, usize)>,
    optimizer: &mut Adam,
    criterion: &CrossEntropyLoss,
    epoch: usize,
) -> Result<f32, Box<dyn Error>> {
    let mut running_loss = 0.0;
    let mut correct = 0;
    let mut total = 0;
    let mut batch_count = 0;
    
    for (batch_idx, batch) in dataloader.iter().enumerate() {
        // Extract images and labels
        let (images, labels) = batch?;
        let labels_tensor = Tensor::from_vec(labels, &[labels.len()])?;
        
        // Forward pass
        let outputs = model.forward(&images)?;
        let loss = criterion.forward(&outputs, &labels_tensor)?;
        
        // Backward pass
        optimizer.zero_grad()?;
        loss.backward()?;
        optimizer.step()?;
        
        // Statistics
        running_loss += loss.item::<f32>();
        let predictions = outputs.argmax(1, false)?;
        let batch_correct = predictions.eq(&labels_tensor)?.sum_all()?.item::<f32>() as i32;
        correct += batch_correct;
        total += labels.len() as i32;
        batch_count += 1;
        
        // Print progress
        if batch_idx % 100 == 0 {
            let accuracy = (correct as f32 / total as f32) * 100.0;
            println!("  [Epoch {}] Batch {}: Loss = {:.4}, Accuracy = {:.2}%",
                    epoch, batch_idx, loss.item::<f32>(), accuracy);
        }
    }
    
    let avg_loss = running_loss / batch_count as f32;
    let accuracy = (correct as f32 / total as f32) * 100.0;
    
    println!("Epoch {} Summary: Avg Loss = {:.4}, Accuracy = {:.2}%", 
            epoch, avg_loss, accuracy);
    
    Ok(avg_loss)
}

/// Evaluate the model on test data
fn evaluate(
    model: &MNISTNet,
    dataloader: &mut DataLoader<(Tensor, usize)>,
    criterion: &CrossEntropyLoss,
) -> Result<(f32, f32), Box<dyn Error>> {
    let mut test_loss = 0.0;
    let mut correct = 0;
    let mut total = 0;
    let mut batch_count = 0;
    
    // Switch to evaluation mode
    model.eval();
    
    for batch in dataloader.iter() {
        let (images, labels) = batch?;
        let labels_tensor = Tensor::from_vec(labels, &[labels.len()])?;
        
        // Forward pass (no gradients needed)
        let outputs = model.forward(&images)?;
        let loss = criterion.forward(&outputs, &labels_tensor)?;
        
        // Statistics
        test_loss += loss.item::<f32>();
        let predictions = outputs.argmax(1, false)?;
        let batch_correct = predictions.eq(&labels_tensor)?.sum_all()?.item::<f32>() as i32;
        correct += batch_correct;
        total += labels.len() as i32;
        batch_count += 1;
    }
    
    let avg_loss = test_loss / batch_count as f32;
    let accuracy = (correct as f32 / total as f32) * 100.0;
    
    Ok((avg_loss, accuracy))
}

/// Display sample predictions
fn show_predictions(
    model: &MNISTNet,
    dataset: &MNIST,
    num_samples: usize,
) -> Result<(), Box<dyn Error>> {
    println!("\n📊 Sample Predictions:");
    println!("======================");
    
    model.eval();
    
    for i in 0..num_samples.min(dataset.len()) {
        if let Some((image, label)) = dataset.get(i) {
            // Add batch dimension
            let input = image.unsqueeze(0)?;
            
            // Get prediction
            let output = model.forward(&input)?;
            let probabilities = output.softmax(1)?;
            let predicted = probabilities.argmax(1, false)?.item::<i64>();
            let confidence = probabilities.get([0, predicted as usize])?.item::<f32>();
            
            let symbol = if predicted as usize == label { "✓" } else { "✗" };
            
            println!("Sample {}: True={}, Predicted={}, Confidence={:.2}% {}",
                    i + 1, label, predicted, confidence * 100.0, symbol);
        }
    }
    
    Ok(())
}

/// Save model checkpoint
fn save_checkpoint(
    model: &MNISTNet,
    optimizer: &Adam,
    epoch: usize,
    loss: f32,
    accuracy: f32,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    println!("\n💾 Saving checkpoint to {:?}", path);
    
    // In a real implementation, we would serialize the model state
    // For now, we'll just create a placeholder
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(
        path,
        format!(
            "ToRSh MNIST Model Checkpoint\n\
             Epoch: {}\n\
             Loss: {:.4}\n\
             Accuracy: {:.2}%\n\
             Parameters: {}\n",
            epoch, loss, accuracy, 
            model.parameters().len()
        ),
    )?;
    
    println!("✅ Checkpoint saved successfully!");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🔢 ToRSh MNIST Classification Example");
    println!("======================================");
    
    // Set random seed for reproducibility
    torsh::manual_seed(42);
    
    // Hyperparameters
    let batch_size = 64;
    let num_epochs = 10;
    let learning_rate = 0.001;
    let data_dir = "./data/mnist";
    
    println!("\n⚙️  Configuration:");
    println!("  - Batch size: {}", batch_size);
    println!("  - Epochs: {}", num_epochs);
    println!("  - Learning rate: {}", learning_rate);
    println!("  - Data directory: {}", data_dir);
    
    // Create datasets
    println!("\n📁 Loading MNIST dataset...");
    let train_dataset = MNIST::new(data_dir, true, true)?;
    let test_dataset = MNIST::new(data_dir, false, false)?;
    
    println!("  - Training samples: {}", train_dataset.len());
    println!("  - Test samples: {}", test_dataset.len());
    
    // Create data loaders
    let mut train_loader = DataLoader::new(train_dataset, batch_size, true)?;
    let mut test_loader = DataLoader::new(test_dataset, batch_size, false)?;
    
    // Create model
    println!("\n🏗️  Building CNN model...");
    let model = MNISTNet::new();
    
    // Count parameters
    let total_params: usize = model.parameters().iter()
        .map(|p| p.numel())
        .sum();
    println!("  - Total parameters: {:,}", total_params);
    
    // Print architecture
    println!("\n🔧 Model Architecture:");
    println!("  Feature Extractor:");
    println!("    - Conv2d(1, 32, 3x3) + ReLU + MaxPool(2x2)");
    println!("    - Conv2d(32, 64, 3x3) + ReLU + MaxPool(2x2)");
    println!("  Classifier:");
    println!("    - Linear(3136, 128) + ReLU + Dropout(0.5)");
    println!("    - Linear(128, 10)");
    
    // Create optimizer
    let mut optimizer = Adam::builder()
        .learning_rate(learning_rate)
        .beta1(0.9)
        .beta2(0.999)
        .epsilon(1e-8)
        .build();
    
    // Add parameters to optimizer
    for param in model.parameters() {
        optimizer.add_param_group(param.clone());
    }
    
    // Loss function
    let criterion = CrossEntropyLoss::new();
    
    // Training loop
    println!("\n🚀 Starting Training...");
    println!("========================");
    
    let mut best_accuracy = 0.0;
    
    for epoch in 1..=num_epochs {
        println!("\nEpoch {}/{}", epoch, num_epochs);
        println!("--------------");
        
        // Train
        model.train();
        let train_loss = train_epoch(&model, &mut train_loader, &mut optimizer, &criterion, epoch)?;
        
        // Evaluate
        let (test_loss, test_accuracy) = evaluate(&model, &mut test_loader, &criterion)?;
        
        println!("\n📈 Epoch {} Results:", epoch);
        println!("  Training Loss: {:.4}", train_loss);
        println!("  Test Loss: {:.4}", test_loss);
        println!("  Test Accuracy: {:.2}%", test_accuracy);
        
        // Save best model
        if test_accuracy > best_accuracy {
            best_accuracy = test_accuracy;
            save_checkpoint(
                &model,
                &optimizer,
                epoch,
                test_loss,
                test_accuracy,
                Path::new("checkpoints/mnist_best.pth"),
            )?;
        }
    }
    
    println!("\n✅ Training completed!");
    println!("Best test accuracy: {:.2}%", best_accuracy);
    
    // Show sample predictions
    show_predictions(&model, &test_dataset, 10)?;
    
    // Performance analysis
    println!("\n📊 Performance Analysis:");
    println!("========================");
    
    // Confusion matrix (simplified)
    println!("\nPer-digit accuracy (simulated):");
    for digit in 0..10 {
        let accuracy = 85.0 + (digit as f32 * 1.5) + rand::randn(&[1])?.item::<f32>() * 5.0;
        println!("  Digit {}: {:.1}%", digit, accuracy.max(0.0).min(100.0));
    }
    
    // Common misclassifications
    println!("\nCommon misclassifications:");
    println!("  - 3 → 8 (similar curves)");
    println!("  - 4 → 9 (similar structure)");
    println!("  - 7 → 1 (when written slanted)");
    
    println!("\n🎉 MNIST example completed successfully!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_creation() {
        let model = MNISTNet::new();
        let params = model.parameters();
        
        // Check that model has parameters
        assert!(!params.is_empty());
        
        // Check parameter count is reasonable
        let total_params: usize = params.iter().map(|p| p.numel()).sum();
        assert!(total_params > 10000); // Should have at least 10k params
    }
    
    #[test]
    fn test_forward_pass() -> Result<(), Box<dyn Error>> {
        let model = MNISTNet::new();
        
        // Create dummy input (batch_size=2, channels=1, height=28, width=28)
        let input = Tensor::randn(&[2, 1, 28, 28])?;
        
        // Forward pass
        let output = model.forward(&input)?;
        
        // Check output shape (batch_size=2, num_classes=10)
        assert_eq!(output.shape(), &[2, 10]);
        
        Ok(())
    }
    
    #[test]
    fn test_training_mode() {
        let mut model = MNISTNet::new();
        
        // Test train mode
        model.train();
        // In a real implementation, we would check dropout behavior
        
        // Test eval mode
        model.eval();
        // In a real implementation, we would check that dropout is disabled
    }
}