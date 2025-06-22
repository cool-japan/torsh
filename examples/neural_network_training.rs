//! Neural network training example
//! 
//! This example demonstrates how to create and train a simple neural network
//! using ToRSh's PyTorch-compatible API.

use torsh::prelude::*;
use torsh::nn::{Module, Linear, Sequential, ReLU, CrossEntropyLoss};
use torsh::optim::{Adam, Optimizer};
use torsh::tensor::Tensor;
use std::error::Error;

/// Simple feedforward neural network for classification
#[derive(Debug)]
struct MLP {
    layers: Sequential,
}

impl MLP {
    /// Create a new Multi-Layer Perceptron
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let layers = Sequential::new()
            .add_module("fc1", Linear::new(input_size, hidden_size, true))
            .add_module("relu1", ReLU::new(false))
            .add_module("fc2", Linear::new(hidden_size, hidden_size, true))
            .add_module("relu2", ReLU::new(false))
            .add_module("fc3", Linear::new(hidden_size, output_size, true));
        
        Self { layers }
    }
}

impl Module for MLP {
    type Error = torsh::TorshError;
    
    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        self.layers.forward(input)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        self.layers.parameters()
    }
    
    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        self.layers.named_parameters()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 ToRSh Neural Network Training Example");
    println!("==========================================");
    
    // Set random seed for reproducibility
    torsh::manual_seed(42);
    
    // Create synthetic dataset
    let batch_size = 64;
    let input_size = 784; // 28x28 flattened images
    let hidden_size = 128;
    let output_size = 10; // 10 classes
    let num_epochs = 10;
    let learning_rate = 0.001;
    
    println!("📊 Dataset Configuration:");
    println!("  - Input size: {}", input_size);
    println!("  - Hidden size: {}", hidden_size);
    println!("  - Output size: {}", output_size);
    println!("  - Batch size: {}", batch_size);
    println!();
    
    // Create model
    let model = MLP::new(input_size, hidden_size, output_size);
    println!("🧠 Model Architecture:");
    println!("  - Layer 1: {} -> {} (+ ReLU)", input_size, hidden_size);
    println!("  - Layer 2: {} -> {} (+ ReLU)", hidden_size, hidden_size);
    println!("  - Layer 3: {} -> {}", hidden_size, output_size);
    println!();
    
    // Print model parameters count
    let total_params: usize = model.parameters().iter()
        .map(|p| p.numel())
        .sum();
    println!("📈 Total parameters: {}", total_params);
    println!();
    
    // Create optimizer
    let mut optimizer = Adam::builder()
        .learning_rate(learning_rate)
        .beta1(0.9)
        .beta2(0.999)
        .epsilon(1e-8)
        .build();
    
    // Add model parameters to optimizer
    for param in model.parameters() {
        optimizer.add_param_group(param.clone());
    }
    
    // Loss function
    let criterion = CrossEntropyLoss::new();
    
    println!("🎯 Training Configuration:");
    println!("  - Optimizer: Adam (lr={}, β1=0.9, β2=0.999)", learning_rate);
    println!("  - Loss function: Cross-Entropy");
    println!("  - Epochs: {}", num_epochs);
    println!();
    
    // Training loop
    println!("🚀 Starting Training...");
    println!("=======================");
    
    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;
        
        // Generate random batch
        let input = Tensor::randn(&[batch_size, input_size])?;
        let target = Tensor::randint(0, output_size as i64, &[batch_size])?;
        
        // Forward pass
        let output = model.forward(&input)?;
        let loss = criterion.forward(&output, &target)?;
        
        // Backward pass
        optimizer.zero_grad()?;
        loss.backward()?;
        optimizer.step()?;
        
        // Calculate accuracy
        let predictions = output.argmax(1, false)?;
        let batch_correct = predictions.eq(&target)?.sum_all()?.item::<f32>() as i32;
        correct += batch_correct;
        total += batch_size as i32;
        
        epoch_loss += loss.item::<f32>();
        
        let accuracy = (correct as f32 / total as f32) * 100.0;
        
        println!("Epoch [{:2}/{}] - Loss: {:.4}, Accuracy: {:.2}%",
                epoch + 1, num_epochs, epoch_loss, accuracy);
    }
    
    println!();
    println!("✅ Training completed!");
    
    // Model evaluation
    println!("🔍 Model Evaluation:");
    println!("===================");
    
    // Generate test batch
    let test_input = Tensor::randn(&[32, input_size])?;
    let test_target = Tensor::randint(0, output_size as i64, &[32])?;
    
    // Evaluate model
    let test_output = model.forward(&test_input)?;
    let test_loss = criterion.forward(&test_output, &test_target)?;
    let test_predictions = test_output.argmax(1, false)?;
    let test_correct = test_predictions.eq(&test_target)?.sum_all()?.item::<f32>() as i32;
    let test_accuracy = (test_correct as f32 / 32.0) * 100.0;
    
    println!("  - Test Loss: {:.4}", test_loss.item::<f32>());
    println!("  - Test Accuracy: {:.2}%", test_accuracy);
    
    // Show sample predictions
    println!();
    println!("📋 Sample Predictions:");
    println!("=====================");
    for i in 0..5 {
        let pred = test_predictions.get(i)?.item::<i64>();
        let actual = test_target.get(i)?.item::<i64>();
        let confidence = test_output.get([i, pred as usize])?.sigmoid()?.item::<f32>();
        
        println!("  Sample {}: Predicted={}, Actual={}, Confidence={:.2}%",
                i + 1, pred, actual, confidence * 100.0);
    }
    
    println!();
    println!("🎉 Example completed successfully!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mlp_creation() {
        let model = MLP::new(784, 128, 10);
        let params = model.parameters();
        
        // Check that model has parameters
        assert!(!params.is_empty());
        
        // Check parameter shapes
        let named_params = model.named_parameters();
        assert!(named_params.iter().any(|(name, _)| name.contains("fc1")));
        assert!(named_params.iter().any(|(name, _)| name.contains("fc2")));
        assert!(named_params.iter().any(|(name, _)| name.contains("fc3")));
    }
    
    #[test]
    fn test_forward_pass() -> Result<(), Box<dyn Error>> {
        let model = MLP::new(784, 128, 10);
        let input = Tensor::randn(&[1, 784])?;
        
        let output = model.forward(&input)?;
        assert_eq!(output.shape(), &[1, 10]);
        
        Ok(())
    }
}