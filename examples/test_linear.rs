//! Test Linear layer

use torsh::prelude::*;
use torsh_nn::modules::Linear;
use torsh_nn::Module;
use torsh_tensor::creation::*;

fn main() -> Result<()> {
    println!("Testing Linear layer...");
    
    // Create a simple linear layer
    println!("Creating Linear layer...");
    let linear = Linear::new(10, 5, true);
    println!("Linear layer created");
    
    // Create input
    println!("Creating input tensor...");
    let input = randn(&[2, 10]); // batch_size=2, in_features=10
    println!("Input shape: {:?}", input.shape().dims());
    
    // Check weight shape
    let params = linear.named_parameters();
    println!("Weight shape: {:?}", params["weight"].read().shape().dims());
    
    // Test matmul directly first
    println!("Testing matmul directly...");
    let weight = params["weight"].read().clone();
    let test_output = input.matmul(&weight)?;
    println!("Direct matmul output shape: {:?}", test_output.shape().dims());
    
    // Forward pass
    println!("Running forward pass...");
    let output = linear.forward(&input)?;
    println!("Output shape: {:?}", output.shape().dims());
    
    assert_eq!(output.shape().dims(), &[2, 5]);
    println!("Linear layer test passed!");
    
    // Test with larger input
    println!("\nTesting with larger input...");
    let large_linear = Linear::new(100, 50, true);
    let large_input = randn(&[4, 100]);
    let large_output = large_linear.forward(&large_input)?;
    println!("Large output shape: {:?}", large_output.shape().dims());
    assert_eq!(large_output.shape().dims(), &[4, 50]);
    
    println!("All tests passed!");
    Ok(())
}