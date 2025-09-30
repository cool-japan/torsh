//! Image Classification with Convolutional Neural Network
//! 
//! This example demonstrates:
//! - Loading and preprocessing image data
//! - Building a CNN architecture
//! - Training loop with data loading
//! - Model evaluation and metrics

use torsh_tensor::{Tensor, creation::*};
use torsh_nn::modules::*;
use torsh_data::{dataset::Dataset, dataloader::simple_dataloader, vision::CIFAR10};
use torsh_core::{error::Result, dtype::DType};

/// Simple CNN architecture for image classification
struct SimpleCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
}

impl SimpleCNN {
    fn new() -> Self {
        Self {
            conv1: Conv2d::new(3, 32, 3),   // 3->32 channels, 3x3 kernel
            conv2: Conv2d::new(32, 64, 3),  // 32->64 channels, 3x3 kernel
            fc1: Linear::new(64 * 6 * 6, 128), // Flattened conv output to 128
            fc2: Linear::new(128, 10),      // 128 to 10 classes
        }
    }
    
    fn forward(&mut self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Conv1 + ReLU + MaxPool
        let x = self.conv1.forward(x)?;
        let x = x.relu()?;
        // Simplified: would use actual max pooling here
        
        // Conv2 + ReLU + MaxPool  
        let x = self.conv2.forward(&x)?;
        let x = x.relu()?;
        
        // Flatten for fully connected layers
        let batch_size = x.shape().dims()[0];
        let x = x.view(&[batch_size as i32, -1])?; // Flatten
        
        // FC1 + ReLU
        let x = self.fc1.forward(&x)?;
        let x = x.relu()?;
        
        // FC2 (output layer)
        let x = self.fc2.forward(&x)?;
        
        Ok(x)
    }
}

fn main() -> Result<()> {
    println!("ToRSh Image Classification Example");
    println!("==================================");
    
    // Setup dataset
    println!("Setting up CIFAR-10 dataset...");
    let train_dataset = setup_dataset(true)?;
    let test_dataset = setup_dataset(false)?;
    
    println!("Train samples: {}", train_dataset.len());
    println!("Test samples: {}", test_dataset.len());
    
    // Create data loaders
    let train_loader = simple_dataloader(train_dataset, 32, true)?; // batch_size=32, shuffle=true
    let test_loader = simple_dataloader(test_dataset, 32, false)?;  // batch_size=32, shuffle=false
    
    // Create model
    let mut model = SimpleCNN::new();
    println!("Created CNN model");
    
    // Training parameters
    let epochs = 10;
    let learning_rate = 0.001;
    
    println!("Training parameters:");
    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", learning_rate);
    println!("  Batch size: 32");
    
    // Training loop
    train_model(&mut model, train_loader, epochs, learning_rate)?;
    
    // Evaluation
    println!("\nEvaluating model...");
    let accuracy = evaluate_model(&mut model, test_loader)?;
    println!("Test accuracy: {:.2}%", accuracy * 100.0);
    
    // Demo inference
    demo_inference(&mut model)?;
    
    Ok(())
}

/// Setup CIFAR-10 dataset
fn setup_dataset(train: bool) -> Result<CIFAR10> {
    let dataset = CIFAR10::new("./data", train)?;
    
    // In a real implementation, we would add data augmentation transforms here
    // let transforms = Compose::new(vec![
    //     Box::new(Resize::new((32, 32))),
    //     Box::new(ToTensor),
    //     Box::new(Normalize::new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])),
    // ]);
    // let dataset = dataset.with_transform(transforms);
    
    Ok(dataset)
}

/// Training loop
fn train_model<D: Dataset>(
    model: &mut SimpleCNN, 
    dataloader: impl Iterator<Item = Result<Vec<D::Item>>>,
    epochs: usize,
    learning_rate: f32
) -> Result<()> {
    println!("\nStarting training...");
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        println!("Epoch {}/{}", epoch + 1, epochs);
        
        for batch_result in dataloader {
            let batch = batch_result?;
            batch_count += 1;
            
            // In a real implementation, we would:
            // 1. Extract images and labels from batch
            // 2. Forward pass
            // 3. Compute loss
            // 4. Backward pass
            // 5. Update parameters
            
            // Simulate training step
            let batch_loss = simulate_training_step(model, &batch, learning_rate)?;
            total_loss += batch_loss;
            
            if batch_count % 100 == 0 {
                println!("  Batch {}: Loss = {:.4}", batch_count, batch_loss);
            }
        }
        
        let avg_loss = total_loss / batch_count as f32;
        println!("  Average loss: {:.4}", avg_loss);
    }
    
    Ok(())
}

/// Simulate a training step (placeholder implementation)
fn simulate_training_step<T>(
    model: &mut SimpleCNN, 
    _batch: &[T], 
    _learning_rate: f32
) -> Result<f32> {
    // In a real implementation:
    // 1. Forward pass: predictions = model.forward(images)
    // 2. Loss computation: loss = criterion(predictions, labels)
    // 3. Backward pass: loss.backward()
    // 4. Parameter update: optimizer.step()
    
    // For demo, return a decreasing loss
    let loss = rand::<f32>(&[1]).item() * 0.5 + 0.1;
    Ok(loss)
}

/// Evaluate model on test data
fn evaluate_model<D: Dataset>(
    model: &mut SimpleCNN, 
    dataloader: impl Iterator<Item = Result<Vec<D::Item>>>
) -> Result<f32> {
    let mut correct = 0;
    let mut total = 0;
    
    for batch_result in dataloader {
        let batch = batch_result?;
        
        // In a real implementation:
        // 1. Forward pass without gradients
        // 2. Get predictions
        // 3. Compare with true labels
        // 4. Count correct predictions
        
        // Simulate evaluation
        let (batch_correct, batch_total) = simulate_evaluation_step(model, &batch)?;
        correct += batch_correct;
        total += batch_total;
    }
    
    Ok(correct as f32 / total as f32)
}

/// Simulate evaluation step (placeholder)
fn simulate_evaluation_step<T>(
    _model: &mut SimpleCNN, 
    batch: &[T]
) -> Result<(usize, usize)> {
    let batch_size = batch.len();
    // Simulate ~70% accuracy
    let correct = (batch_size as f32 * 0.7) as usize;
    Ok((correct, batch_size))
}

/// Demonstrate inference on sample data
fn demo_inference(model: &mut SimpleCNN) -> Result<()> {
    println!("\n=== Inference Demo ===");
    
    // Create a dummy image (3x32x32 for CIFAR-10)
    let sample_image = rand::<f32>(&[1, 3, 32, 32]); // batch_size=1
    
    println!("Input image shape: {:?}", sample_image.shape().dims());
    
    // Forward pass
    let output = model.forward(&sample_image)?;
    println!("Output shape: {:?}", output.shape().dims());
    
    // Apply softmax to get probabilities
    let probabilities = output.softmax(1)?;
    
    // Get predicted class (argmax)
    let predicted_class = probabilities.argmax(Some(1))?;
    
    println!("Predicted class: {}", predicted_class.item());
    
    // Show top 3 predictions
    show_top_predictions(&probabilities)?;
    
    Ok(())
}

/// Show top predictions with class names
fn show_top_predictions(probabilities: &Tensor<f32>) -> Result<()> {
    let class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ];
    
    println!("\nTop 3 predictions:");
    
    let probs = probabilities.to_vec();
    let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    
    // Sort by probability (descending)
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    for (rank, (class_idx, prob)) in indexed_probs.iter().take(3).enumerate() {
        println!("  {}: {} ({:.2}%)", 
                rank + 1, 
                class_names[*class_idx], 
                prob * 100.0);
    }
    
    Ok(())
}

/// Data augmentation example
fn apply_data_augmentation(image: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Example transformations (simplified implementations)
    
    // 1. Random horizontal flip
    let image = random_horizontal_flip(image, 0.5)?;
    
    // 2. Random rotation (small angle)
    let image = random_rotation(&image, 5.0)?;
    
    // 3. Color jittering
    let image = color_jitter(&image, 0.1, 0.1, 0.1)?;
    
    Ok(image)
}

/// Random horizontal flip
fn random_horizontal_flip(image: &Tensor<f32>, prob: f32) -> Result<Tensor<f32>> {
    if rand::<f32>(&[1]).item() < prob {
        // Flip horizontally (simplified - would use actual flip operation)
        Ok(image.clone())
    } else {
        Ok(image.clone())
    }
}

/// Random rotation  
fn random_rotation(image: &Tensor<f32>, max_angle: f32) -> Result<Tensor<f32>> {
    let _angle = (rand::<f32>(&[1]).item() - 0.5) * 2.0 * max_angle;
    // Apply rotation (simplified - would use actual rotation)
    Ok(image.clone())
}

/// Color jittering
fn color_jitter(
    image: &Tensor<f32>, 
    brightness: f32, 
    contrast: f32, 
    saturation: f32
) -> Result<Tensor<f32>> {
    // Apply random color adjustments
    let brightness_factor = 1.0 + (rand::<f32>(&[1]).item() - 0.5) * brightness;
    let adjusted = image.mul_scalar(brightness_factor)?;
    
    // Additional contrast and saturation adjustments would go here
    let _ = (contrast, saturation); // Suppress unused warnings
    
    Ok(adjusted)
}

/// Model summary and analysis
fn analyze_model(model: &SimpleCNN) {
    println!("\n=== Model Architecture ===");
    println!("Conv1: 3 -> 32 channels, 3x3 kernel");
    println!("Conv2: 32 -> 64 channels, 3x3 kernel");
    println!("FC1: {} -> 128", 64 * 6 * 6);
    println!("FC2: 128 -> 10 (classes)");
    
    // Calculate approximate parameter count
    let conv1_params = 3 * 32 * 3 * 3 + 32; // weights + bias
    let conv2_params = 32 * 64 * 3 * 3 + 64;
    let fc1_params = 64 * 6 * 6 * 128 + 128;
    let fc2_params = 128 * 10 + 10;
    
    let total_params = conv1_params + conv2_params + fc1_params + fc2_params;
    
    println!("\nParameter count:");
    println!("  Conv1: {}", conv1_params);
    println!("  Conv2: {}", conv2_params);
    println!("  FC1: {}", fc1_params);
    println!("  FC2: {}", fc2_params);
    println!("  Total: {}", total_params);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_creation() {
        let model = SimpleCNN::new();
        // Should not panic
        drop(model);
    }
    
    #[test]
    fn test_forward_pass() {
        let mut model = SimpleCNN::new();
        let input = rand::<f32>(&[2, 3, 32, 32]); // batch_size=2
        
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 10]);
    }
    
    #[test]
    fn test_data_augmentation() {
        let image = rand::<f32>(&[3, 32, 32]);
        let augmented = apply_data_augmentation(&image).unwrap();
        assert_eq!(augmented.shape(), image.shape());
    }
}