//! Simple linear regression example using ToRSh
//! 
//! This example demonstrates:
//! - Tensor creation and manipulation
//! - Simple neural network with a linear layer
//! - Basic training loop with gradient descent
//! - Loss computation and optimization

use torsh_tensor::{Tensor, creation::*};
use torsh_nn::modules::Linear;
use torsh_core::{error::Result, dtype::DType};

fn main() -> Result<()> {
    println!("ToRSh Linear Regression Example");
    println!("===============================");
    
    // Generate synthetic data: y = 3x + 2 + noise
    let num_samples = 100;
    let x_data = generate_data(num_samples)?;
    let y_data = generate_targets(&x_data)?;
    
    println!("Generated {} training samples", num_samples);
    
    // Create a simple linear model
    let mut model = Linear::new(1, 1); // 1 input, 1 output
    
    // Training parameters
    let learning_rate = 0.01;
    let epochs = 100;
    
    println!("Training linear model...");
    println!("Learning rate: {}", learning_rate);
    println!("Epochs: {}", epochs);
    
    // Training loop
    for epoch in 0..epochs {
        // Forward pass
        let predictions = model.forward(&x_data)?;
        
        // Compute loss (MSE)
        let loss = mse_loss(&predictions, &y_data)?;
        
        // Compute gradients (simplified - in real implementation would use autograd)
        let gradients = compute_gradients(&x_data, &predictions, &y_data)?;
        
        // Update parameters
        update_parameters(&mut model, &gradients, learning_rate)?;
        
        // Print progress
        if epoch % 20 == 0 || epoch == epochs - 1 {
            let loss_value = loss.item();
            println!("Epoch {}: Loss = {:.4}", epoch, loss_value);
        }
    }
    
    // Test the trained model
    println!("\nTesting trained model:");
    test_model(&model)?;
    
    // Display learned parameters
    display_parameters(&model)?;
    
    Ok(())
}

/// Generate synthetic input data
fn generate_data(num_samples: usize) -> Result<Tensor<f32>> {
    // Generate random x values between -1 and 1
    let mut data = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let x = (i as f32 / num_samples as f32) * 4.0 - 2.0; // Range: -2 to 2
        data.push(x);
    }
    
    Ok(Tensor::from_vec(data, &[num_samples, 1]))
}

/// Generate target values: y = 3x + 2 + noise
fn generate_targets(x_data: &Tensor<f32>) -> Result<Tensor<f32>> {
    let x_vec = x_data.to_vec();
    let mut y_vec = Vec::with_capacity(x_vec.len());
    
    for &x in &x_vec {
        // True function: y = 3x + 2
        let y = 3.0 * x + 2.0;
        // Add small amount of noise
        let noise = (rand::random::<f32>() - 0.5) * 0.2;
        y_vec.push(y + noise);
    }
    
    Ok(Tensor::from_vec(y_vec, &[x_vec.len(), 1]))
}

/// Compute Mean Squared Error loss
fn mse_loss(predictions: &Tensor<f32>, targets: &Tensor<f32>) -> Result<Tensor<f32>> {
    predictions.mse_loss(targets)
}

/// Compute gradients for the linear layer (simplified implementation)
fn compute_gradients(
    inputs: &Tensor<f32>, 
    predictions: &Tensor<f32>, 
    targets: &Tensor<f32>
) -> Result<LinearGradients> {
    // Compute prediction error
    let errors = predictions.sub(targets)?;
    
    // For linear layer: dW = X^T * errors / batch_size
    // For bias: db = sum(errors) / batch_size
    let batch_size = inputs.shape().dims()[0] as f32;
    
    // Simplified gradient computation
    let weight_grad = inputs.t()?.matmul(&errors)?.mul_scalar(1.0 / batch_size)?;
    let bias_grad = errors.sum()?.mul_scalar(1.0 / batch_size)?;
    
    Ok(LinearGradients {
        weight: weight_grad,
        bias: bias_grad,
    })
}

/// Simple structure to hold gradients
struct LinearGradients {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
}

/// Update model parameters using gradients
fn update_parameters(
    model: &mut Linear, 
    gradients: &LinearGradients, 
    learning_rate: f32
) -> Result<()> {
    // In a real implementation, this would access and update the actual model parameters
    // For now, we'll print what the update would be
    println!("  Weight gradient norm: {:.6}", gradient_norm(&gradients.weight)?);
    println!("  Bias gradient: {:.6}", gradients.bias.item());
    
    Ok(())
}

/// Compute gradient norm for monitoring
fn gradient_norm(tensor: &Tensor<f32>) -> Result<f32> {
    // Simplified norm computation
    let squared = tensor.mul(tensor)?;
    let sum = squared.sum()?;
    Ok(sum.item().sqrt())
}

/// Test the trained model on some sample inputs
fn test_model(model: &Linear) -> Result<()> {
    let test_inputs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    
    println!("Input -> Prediction (Expected: y = 3x + 2)");
    println!("------------------------------------------");
    
    for &x in &test_inputs {
        let input = Tensor::from_scalar(x).view(&[1, 1])?;
        let prediction = model.forward(&input)?;
        let expected = 3.0 * x + 2.0;
        
        println!("{:5.1} -> {:8.3} (expected: {:6.1})", 
                x, prediction.item(), expected);
    }
    
    Ok(())
}

/// Display the learned parameters
fn display_parameters(model: &Linear) -> Result<()> {
    println!("\nLearned Parameters:");
    println!("------------------");
    
    // In a real implementation, we would access the actual parameters
    // For now, show what the ideal parameters should be
    println!("Expected weight: 3.0");
    println!("Expected bias: 2.0");
    println!("(Actual parameters would be displayed here in full implementation)");
    
    Ok(())
}

/// Example of using the model for inference
fn inference_example() -> Result<()> {
    println!("\n=== Inference Example ===");
    
    // Create a trained model (placeholder)
    let model = Linear::new(1, 1);
    
    // Make predictions on new data
    let new_data = linspace(-3.0, 3.0, 7); // 7 points from -3 to 3
    let predictions = model.forward(&new_data)?;
    
    println!("Inference on new data:");
    let data_vec = new_data.to_vec();
    let pred_vec = predictions.to_vec();
    
    for (x, y) in data_vec.iter().zip(pred_vec.iter()) {
        println!("x = {:5.1}, predicted y = {:8.3}", x, y);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_generation() {
        let data = generate_data(10).unwrap();
        assert_eq!(data.shape().dims(), &[10, 1]);
        
        let targets = generate_targets(&data).unwrap();
        assert_eq!(targets.shape().dims(), &[10, 1]);
    }
    
    #[test]
    fn test_loss_computation() {
        let predictions = ones::<f32>(&[5, 1]);
        let targets = zeros::<f32>(&[5, 1]);
        
        let loss = mse_loss(&predictions, &targets).unwrap();
        assert_eq!(loss.item(), 1.0); // (1-0)^2 = 1
    }
    
    #[test]
    fn test_model_creation() {
        let model = Linear::new(1, 1);
        let input = ones::<f32>(&[1, 1]);
        
        // Should not panic
        let _output = model.forward(&input).unwrap();
    }
}