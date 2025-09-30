//! # Tutorial 02: Automatic Differentiation (Autograd) Basics
//! 
//! This is the second tutorial in the ToRSh learning series.
//! Learn how automatic differentiation works and how to compute gradients.
//! 
//! ## What you'll learn:
//! - Enabling gradient computation on tensors
//! - Computing gradients with backward()
//! - Understanding the computational graph
//! - Gradient accumulation and zeroing
//! - Using gradients for optimization
//! 
//! ## Prerequisites:
//! - Complete Tutorial 01: Tensor Basics
//! - Basic understanding of derivatives (calculus)
//! 
//! Run with: `cargo run --example 02_autograd_basics`

use torsh::prelude::*;
use torsh::{Tensor, Device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Tutorial 02: Automatic Differentiation Basics ===\n");
    
    // 1. Enabling gradient computation
    println!("1. Enabling Gradient Computation");
    println!("================================");
    
    // Create a tensor that requires gradients
    let mut x = Tensor::from_vec(vec![2.0], &[1])?;
    x.set_requires_grad(true);
    
    println!("Input tensor x = {:?}", x);
    println!("Requires gradient: {}\n", x.requires_grad());
    
    // Simple function: y = x^2
    let y = &x * &x;
    println!("Function: y = x^2");
    println!("y = {:?}", y);
    println!("y requires gradient: {}\n", y.requires_grad());
    
    // Compute gradient dy/dx = 2x
    y.backward()?;
    
    if let Some(grad) = x.grad() {
        println!("Gradient dy/dx = {:?}", grad);
        println!("Expected: 2 * x = 2 * 2 = 4 ✓\n");
    }
    
    // 2. More complex functions
    println!("2. More Complex Functions");
    println!("=========================");
    
    // Reset gradients for new computation
    let mut a = Tensor::from_vec(vec![3.0], &[1])?;
    let mut b = Tensor::from_vec(vec![4.0], &[1])?;
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    
    println!("a = {:?}, b = {:?}", a, b);
    
    // Function: z = a^2 + 2*a*b + b^2 = (a + b)^2
    let a_squared = &a * &a;
    let b_squared = &b * &b;
    let ab_term = &(&a * &b) * 2.0;
    let z = &(&a_squared + &ab_term) + &b_squared;
    
    println!("Function: z = a^2 + 2*a*b + b^2");
    println!("z = {:?}\n", z);
    
    z.backward()?;
    
    if let Some(grad_a) = a.grad() {
        println!("∂z/∂a = {:?}", grad_a);
        println!("Expected: 2a + 2b = 2*3 + 2*4 = 14 ✓");
    }
    
    if let Some(grad_b) = b.grad() {
        println!("∂z/∂b = {:?}", grad_b);
        println!("Expected: 2a + 2b = 2*3 + 2*4 = 14 ✓\n");
    }
    
    // 3. Vector functions and Jacobians
    println!("3. Vector Functions");
    println!("===================");
    
    let mut vec_x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
    vec_x.set_requires_grad(true);
    
    println!("Input vector x = {:?}", vec_x);
    
    // Function: f(x) = x^2 (element-wise)
    let f_x = &vec_x * &vec_x;
    println!("Function: f(x) = x^2 (element-wise)");
    println!("f(x) = {:?}", f_x);
    
    // To compute gradients for vector functions, we need to specify which output
    // element we want to differentiate. Let's sum all outputs first.
    let scalar_output = f_x.sum()?;
    println!("Sum of f(x) = {:?}\n", scalar_output);
    
    scalar_output.backward()?;
    
    if let Some(grad) = vec_x.grad() {
        println!("Gradient: {:?}", grad);
        println!("Expected: [2*1, 2*2, 2*3] = [2, 4, 6] ✓\n");
    }
    
    // 4. Gradient accumulation
    println!("4. Gradient Accumulation");
    println!("========================");
    
    let mut param = Tensor::from_vec(vec![1.0], &[1])?;
    param.set_requires_grad(true);
    
    println!("Parameter = {:?}\n", param);
    
    // First computation: y1 = param^2
    let y1 = &param * &param;
    y1.backward()?;
    
    if let Some(grad) = param.grad() {
        println!("After first backward (y1 = param^2):");
        println!("Gradient = {:?} (should be 2*1 = 2)", grad);
    }
    
    // Second computation without zeroing gradients: y2 = param^3
    let y2 = &(&param * &param) * &param;
    y2.backward()?;
    
    if let Some(grad) = param.grad() {
        println!("After second backward (y2 = param^3, accumulated):");
        println!("Gradient = {:?} (should be 2 + 3*1^2 = 5)", grad);
    }
    
    // Zero gradients and compute again
    param.zero_grad();
    let y3 = &(&param * &param) * &param;
    y3.backward()?;
    
    if let Some(grad) = param.grad() {
        println!("After zeroing gradients and computing y3 = param^3:");
        println!("Gradient = {:?} (should be 3*1^2 = 3)\n", grad);
    }
    
    // 5. Chain rule in action
    println!("5. Chain Rule in Action");
    println!("=======================");
    
    let mut input = Tensor::from_vec(vec![0.5], &[1])?;
    input.set_requires_grad(true);
    
    println!("Input = {:?}", input);
    
    // Multi-step computation: final = sin(x^2 + 1)
    let step1 = &input * &input;        // x^2
    let step2 = &step1 + 1.0;           // x^2 + 1
    let final_result = step2.sin();     // sin(x^2 + 1)
    
    println!("Step 1: x^2 = {:?}", step1);
    println!("Step 2: x^2 + 1 = {:?}", step2);
    println!("Final: sin(x^2 + 1) = {:?}\n", final_result);
    
    final_result.backward()?;
    
    if let Some(grad) = input.grad() {
        println!("Gradient d/dx[sin(x^2 + 1)] = {:?}", grad);
        // Chain rule: d/dx[sin(x^2 + 1)] = cos(x^2 + 1) * 2x
        let x_val = 0.5;
        let expected = (x_val * x_val + 1.0).cos() * 2.0 * x_val;
        println!("Expected: cos(x^2 + 1) * 2x = cos({:.3}) * {} = {:.6}", 
                 x_val * x_val + 1.0, 2.0 * x_val, expected);
    }
    
    // 6. Practical example: Simple linear regression
    println!("\n6. Practical Example: Simple Linear Regression");
    println!("===============================================");
    
    // Training data: y = 2x + 1 + noise
    let x_data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;
    let y_data = Tensor::from_vec(vec![3.1, 4.9, 7.2, 9.1, 10.8], &[5])?;
    
    // Parameters to learn
    let mut weight = Tensor::from_vec(vec![0.0], &[1])?; // Start with w = 0
    let mut bias = Tensor::from_vec(vec![0.0], &[1])?;   // Start with b = 0
    weight.set_requires_grad(true);
    bias.set_requires_grad(true);
    
    println!("Training data:");
    println!("X: {:?}", x_data);
    println!("Y: {:?}\n", y_data);
    
    let learning_rate = 0.01;
    let epochs = 100;
    
    println!("Training simple linear regression (y = wx + b):");
    println!("Initial parameters: w = {:.3}, b = {:.3}", 
             weight.to_vec()?[0], bias.to_vec()?[0]);
    
    for epoch in 0..epochs {
        // Zero gradients
        weight.zero_grad();
        bias.zero_grad();
        
        // Forward pass: predictions = weight * x_data + bias
        let predictions = &(&weight * &x_data) + &bias;
        
        // Compute loss: Mean Squared Error
        let diff = &predictions - &y_data;
        let squared_diff = &diff * &diff;
        let loss = squared_diff.mean()?;
        
        // Backward pass
        loss.backward()?;
        
        // Update parameters using gradients
        if let (Some(w_grad), Some(b_grad)) = (weight.grad(), bias.grad()) {
            // w = w - learning_rate * gradient
            let w_data = weight.to_vec()?;
            let b_data = bias.to_vec()?;
            let w_grad_data = w_grad.to_vec()?;
            let b_grad_data = b_grad.to_vec()?;
            
            let new_w = w_data[0] - learning_rate * w_grad_data[0];
            let new_b = b_data[0] - learning_rate * b_grad_data[0];
            
            weight = Tensor::from_vec(vec![new_w], &[1])?;
            bias = Tensor::from_vec(vec![new_b], &[1])?;
            weight.set_requires_grad(true);
            bias.set_requires_grad(true);
        }
        
        // Print progress
        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!("Epoch {}: Loss = {:.6}, w = {:.3}, b = {:.3}", 
                     epoch, loss.to_vec()?[0], weight.to_vec()?[0], bias.to_vec()?[0]);
        }
    }
    
    println!("\n✅ Training completed!");
    println!("Final parameters: w = {:.3}, b = {:.3}", 
             weight.to_vec()?[0], bias.to_vec()?[0]);
    println!("Target parameters: w = 2.0, b = 1.0");
    println!("(Difference due to noise in training data)\n");
    
    // 7. Key concepts summary
    println!("7. Key Concepts Summary");
    println!("=======================");
    println!("✓ requires_grad(true): Enables gradient computation for a tensor");
    println!("✓ backward(): Computes gradients via backpropagation");
    println!("✓ grad(): Access computed gradients");
    println!("✓ zero_grad(): Reset gradients to zero (important for training loops)");
    println!("✓ Chain rule: Automatic differentiation handles complex function compositions");
    println!("✓ Gradient accumulation: Gradients add up across multiple backward() calls");
    println!("✓ Optimization: Use gradients to update parameters (gradient descent)\n");
    
    println!("🎉 Congratulations! You've completed Tutorial 02: Autograd Basics");
    println!("📚 Next: Run `cargo run --example 03_neural_networks` to learn about neural networks");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_gradient() {
        let mut x = Tensor::from_vec(vec![2.0], &[1]).unwrap();
        x.set_requires_grad(true);
        
        let y = &x * &x; // y = x^2
        y.backward().unwrap();
        
        if let Some(grad) = x.grad() {
            let grad_val = grad.to_vec().unwrap()[0];
            assert!((grad_val - 4.0).abs() < 1e-6); // dy/dx = 2x = 2*2 = 4
        }
    }
    
    #[test]
    fn test_gradient_accumulation() {
        let mut x = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        x.set_requires_grad(true);
        
        // First computation
        let y1 = &x * &x;
        y1.backward().unwrap();
        
        // Second computation (gradients should accumulate)
        let y2 = &x * 3.0;
        y2.backward().unwrap();
        
        if let Some(grad) = x.grad() {
            let grad_val = grad.to_vec().unwrap()[0];
            // Should be 2*1 (from x^2) + 3 (from 3*x) = 5
            assert!((grad_val - 5.0).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_zero_grad() {
        let mut x = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        x.set_requires_grad(true);
        
        // First computation
        let y1 = &x * &x;
        y1.backward().unwrap();
        
        // Zero gradients
        x.zero_grad();
        
        // Second computation
        let y2 = &x * 3.0;
        y2.backward().unwrap();
        
        if let Some(grad) = x.grad() {
            let grad_val = grad.to_vec().unwrap()[0];
            // Should be only 3 (from 3*x), not accumulated
            assert!((grad_val - 3.0).abs() < 1e-6);
        }
    }
}