//! # Tutorial 03: Neural Networks
//! 
//! This is the third tutorial in the ToRSh learning series.
//! Learn how to build, train, and use neural networks with ToRSh.
//! 
//! ## What you'll learn:
//! - Creating neural network layers (Linear, Activation)
//! - Forward and backward passes
//! - Loss functions and optimization
//! - Training a complete neural network
//! - Making predictions with trained models
//! 
//! ## Prerequisites:
//! - Complete Tutorial 01: Tensor Basics
//! - Complete Tutorial 02: Autograd Basics
//! - Understanding of basic machine learning concepts
//! 
//! Run with: `cargo run --example 03_neural_networks`

use torsh::prelude::*;
use torsh::{Tensor, Device};
use torsh::nn::{Module, Linear, ReLU, MSELoss};
use std::f32::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Tutorial 03: Neural Networks ===\n");
    
    // 1. Understanding neural network layers
    println!("1. Neural Network Layers");
    println!("========================");
    
    // Create a simple linear layer: y = Wx + b
    let mut linear_layer = Linear::new(3, 2)?; // 3 inputs, 2 outputs
    
    println!("Linear layer: 3 inputs → 2 outputs");
    if let Some(weight) = linear_layer.weight() {
        println!("Weight shape: {:?}", weight.shape());
    }
    if let Some(bias) = linear_layer.bias() {
        println!("Bias shape: {:?}", bias.shape());
    }
    
    // Forward pass through the layer
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3])?; // Batch size 1, 3 features
    let output = linear_layer.forward(&input)?;
    
    println!("Input shape: {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());
    println!("Input: {:?}", input);
    println!("Output: {:?}\n", output);
    
    // 2. Activation functions
    println!("2. Activation Functions");
    println!("=======================");
    
    let relu = ReLU::new();
    let test_input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5])?;
    let relu_output = relu.forward(&test_input)?;
    
    println!("ReLU activation function:");
    println!("Input:  {:?}", test_input);
    println!("Output: {:?}", relu_output);
    println!("ReLU sets negative values to 0, keeps positive values unchanged\n");
    
    // Manual activation functions
    let sigmoid_input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5])?;
    let sigmoid_output = sigmoid_input.sigmoid();
    
    println!("Sigmoid activation function:");
    println!("Input:  {:?}", sigmoid_input);
    println!("Output: {:?}", sigmoid_output);
    println!("Sigmoid maps values to (0, 1) range\n");
    
    // 3. Building a multi-layer network
    println!("3. Multi-Layer Neural Network");
    println!("=============================");
    
    // Create a simple 2-layer network for binary classification
    struct SimpleNet {
        layer1: Linear,
        relu: ReLU,
        layer2: Linear,
    }
    
    impl SimpleNet {
        fn new() -> Result<Self, TorshError> {
            Ok(Self {
                layer1: Linear::new(2, 4)?, // 2 inputs, 4 hidden units
                relu: ReLU::new(),
                layer2: Linear::new(4, 1)?, // 4 hidden units, 1 output
            })
        }
        
        fn forward(&mut self, x: &Tensor) -> Result<Tensor, TorshError> {
            let h1 = self.layer1.forward(x)?;
            let h1_activated = self.relu.forward(&h1)?;
            let output = self.layer2.forward(&h1_activated)?;
            Ok(output)
        }
        
        fn parameters(&mut self) -> Vec<&mut Tensor> {
            let mut params = Vec::new();
            params.extend(self.layer1.parameters());
            params.extend(self.layer2.parameters());
            params
        }
    }
    
    let mut network = SimpleNet::new()?;
    
    // Test forward pass
    let test_input = Tensor::from_vec(vec![0.5, -0.3], &[1, 2])?;
    let prediction = network.forward(&test_input)?;
    
    println!("Network architecture: 2 → 4 → 1");
    println!("Test input: {:?}", test_input);
    println!("Network output: {:?}\n", prediction);
    
    // 4. Training data generation (XOR problem)
    println!("4. Training Data: XOR Problem");
    println!("=============================");
    
    // XOR truth table: A XOR B = (A AND !B) OR (!A AND B)
    let training_inputs = Tensor::from_vec(
        vec![
            0.0, 0.0,  // 0 XOR 0 = 0
            0.0, 1.0,  // 0 XOR 1 = 1
            1.0, 0.0,  // 1 XOR 0 = 1
            1.0, 1.0,  // 1 XOR 1 = 0
        ],
        &[4, 2] // 4 samples, 2 features each
    )?;
    
    let training_targets = Tensor::from_vec(
        vec![0.0, 1.0, 1.0, 0.0], // Expected XOR outputs
        &[4, 1] // 4 samples, 1 output each
    )?;
    
    println!("XOR Training Data:");
    println!("Inputs:  {:?}", training_inputs);
    println!("Targets: {:?}", training_targets);
    println!("This is a classic non-linearly separable problem\n");
    
    // 5. Loss function
    println!("5. Loss Function");
    println!("================");
    
    let loss_fn = MSELoss::new();
    
    // Example loss calculation
    let example_predictions = Tensor::from_vec(vec![0.2, 0.8, 0.7, 0.1], &[4, 1])?;
    let example_loss = loss_fn.forward(&example_predictions, &training_targets)?;
    
    println!("Mean Squared Error (MSE) Loss Function:");
    println!("Predictions: {:?}", example_predictions);
    println!("Targets:     {:?}", training_targets);
    println!("Loss:        {:?}", example_loss);
    println!("Lower loss = better predictions\n");
    
    // 6. Training loop
    println!("6. Training the Neural Network");
    println!("==============================");
    
    let learning_rate = 0.1;
    let epochs = 1000;
    
    println!("Training XOR neural network...");
    println!("Learning rate: {}", learning_rate);
    println!("Epochs: {}\n", epochs);
    
    for epoch in 0..epochs {
        // Zero gradients
        for param in network.parameters() {
            param.zero_grad();
        }
        
        // Forward pass
        let predictions = network.forward(&training_inputs)?;
        
        // Compute loss
        let loss = loss_fn.forward(&predictions, &training_targets)?;
        
        // Backward pass
        loss.backward()?;
        
        // Update parameters using gradient descent
        for param in network.parameters() {
            if let Some(grad) = param.grad() {
                let param_data = param.to_vec()?;
                let grad_data = grad.to_vec()?;
                
                let updated_data: Vec<f32> = param_data.iter()
                    .zip(grad_data.iter())
                    .map(|(&p, &g)| p - learning_rate * g)
                    .collect();
                
                *param = Tensor::from_vec(updated_data, param.shape().dims())?;
                param.set_requires_grad(true);
            }
        }
        
        // Print progress
        if epoch % 200 == 0 || epoch == epochs - 1 {
            let loss_val = loss.to_vec()?[0];
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
            
            // Show current predictions
            if epoch == epochs - 1 {
                let final_predictions = network.forward(&training_inputs)?;
                println!("Final predictions: {:?}", final_predictions);
                println!("Targets:          {:?}", training_targets);
            }
        }
    }
    
    // 7. Testing the trained network
    println!("\n7. Testing the Trained Network");
    println!("==============================");
    
    // Test each XOR combination
    let test_cases = vec![
        (vec![0.0, 0.0], "0 XOR 0"),
        (vec![0.0, 1.0], "0 XOR 1"),
        (vec![1.0, 0.0], "1 XOR 0"),
        (vec![1.0, 1.0], "1 XOR 1"),
    ];
    
    println!("Testing XOR function:");
    for (input_vals, description) in test_cases {
        let test_input = Tensor::from_vec(input_vals.clone(), &[1, 2])?;
        let prediction = network.forward(&test_input)?;
        let pred_val = prediction.to_vec()?[0];
        let rounded = if pred_val > 0.5 { 1 } else { 0 };
        
        println!("{}: Input {:?} → Prediction {:.3} → Rounded {}",
                 description, input_vals, pred_val, rounded);
    }
    
    // 8. Understanding what the network learned
    println!("\n8. Understanding the Network");
    println!("============================");
    
    // Visualize decision boundary (simplified)
    println!("The network learned to separate the XOR function by:");
    println!("1. First layer: Creates features that can distinguish the patterns");
    println!("2. ReLU activation: Introduces non-linearity (essential for XOR)");
    println!("3. Second layer: Combines features to produce final classification");
    println!("\nWithout the hidden layer and ReLU, this problem would be impossible!");
    println!("This demonstrates why deep networks can solve complex problems.\n");
    
    // 9. Saving and loading models (conceptual)
    println!("9. Model Persistence (Conceptual)");
    println!("==================================");
    println!("In practice, you would:");
    println!("✓ Save model parameters after training");
    println!("✓ Load parameters to restore a trained model");
    println!("✓ Use the model for inference on new data");
    println!("✓ Continue training from a checkpoint\n");
    
    // 10. Key concepts summary
    println!("10. Key Neural Network Concepts");
    println!("===============================");
    println!("✓ Layers: Transform input data (Linear, Convolution, etc.)");
    println!("✓ Activation Functions: Add non-linearity (ReLU, Sigmoid, Tanh)");
    println!("✓ Forward Pass: Data flows through network to produce predictions");
    println!("✓ Loss Function: Measures how wrong predictions are");
    println!("✓ Backward Pass: Computes gradients via backpropagation");
    println!("✓ Optimization: Updates parameters using gradients (SGD, Adam, etc.)");
    println!("✓ Training Loop: Repeat forward→loss→backward→update cycle");
    println!("✓ Non-linearity: Essential for learning complex patterns (like XOR)\n");
    
    println!("🎉 Congratulations! You've completed Tutorial 03: Neural Networks");
    println!("📚 Next: Run `cargo run --example 04_cnn_basics` to learn about Convolutional Neural Networks");
    
    Ok(())
}

/// Helper function for creating synthetic classification data
fn create_spiral_data(n_samples: usize, n_classes: usize) -> Result<(Tensor, Tensor), TorshError> {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for class_id in 0..n_classes {
        for i in 0..n_samples {
            let r = i as f32 / n_samples as f32;
            let t = (class_id as f32 * 4.0) + (r * 4.0) + 
                    (rand::random::<f32>() - 0.5) * 0.2;
            
            let x = r * t.cos();
            let y = r * t.sin();
            
            inputs.extend_from_slice(&[x, y]);
            targets.push(class_id as f32);
        }
    }
    
    let input_tensor = Tensor::from_vec(inputs, &[n_samples * n_classes, 2])?;
    let target_tensor = Tensor::from_vec(targets, &[n_samples * n_classes])?;
    
    Ok((input_tensor, target_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_layer() {
        let mut layer = Linear::new(3, 2).unwrap();
        let input = Tensor::ones(&[1, 3]).unwrap();
        let output = layer.forward(&input).unwrap();
        
        assert_eq!(output.shape().dims(), &[1, 2]);
    }
    
    #[test]
    fn test_relu_activation() {
        let relu = ReLU::new();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
        let output = relu.forward(&input).unwrap();
        
        let output_data = output.to_vec().unwrap();
        assert_eq!(output_data[0], 0.0); // -1.0 → 0.0
        assert_eq!(output_data[1], 0.0); // 0.0 → 0.0
        assert_eq!(output_data[2], 1.0); // 1.0 → 1.0
    }
    
    #[test]
    fn test_mse_loss() {
        let loss_fn = MSELoss::new();
        let predictions = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let targets = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let loss = loss_fn.forward(&predictions, &targets).unwrap();
        
        // Perfect predictions should have zero loss
        let loss_val = loss.to_vec().unwrap()[0];
        assert!(loss_val < 1e-6);
    }
}