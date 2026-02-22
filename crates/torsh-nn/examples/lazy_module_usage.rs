//! Comprehensive example demonstrating lazy module usage patterns
//!
//! This example shows how to properly use lazy modules that infer input dimensions
//! from the first forward pass. Lazy modules are useful when you don't know the
//! exact input dimensions at model creation time.

use torsh_core::error::Result;
use torsh_nn::layers::lazy::{LazyConv1d, LazyConv2d, LazyLinear, LazyModule};
use torsh_nn::Module;
use torsh_tensor::creation::randn;

fn main() -> Result<()> {
    println!("=== Lazy Module Usage Examples ===\n");

    // Example 1: LazyLinear usage
    lazy_linear_example()?;

    // Example 2: LazyConv1d usage
    lazy_conv1d_example()?;

    // Example 3: LazyConv2d usage
    lazy_conv2d_example()?;

    // Example 4: Error handling when forgetting to initialize
    error_handling_example()?;

    // Example 5: Sequential network with lazy modules
    sequential_lazy_example()?;

    Ok(())
}

fn lazy_linear_example() -> Result<()> {
    println!("1. LazyLinear Example:");
    println!("   - Creates a linear layer without specifying input features");
    println!("   - Automatically infers input_features from first forward pass");

    // Create a lazy linear layer that will output 128 features
    let mut lazy_linear = LazyLinear::new(128, true);

    // Before initialization, the layer has no parameters
    println!(
        "   - Before initialization: {} parameters",
        lazy_linear.parameters().len()
    );
    assert!(!lazy_linear.is_initialized());

    // Create input tensor with unknown input size (e.g., 64 features)
    let input = randn::<f32>(&[32, 64])?; // [batch_size, input_features]

    // Initialize the layer with the input tensor
    lazy_linear.initialize_lazy(&input)?;

    // After initialization, the layer has weight and bias parameters
    println!(
        "   - After initialization: {} parameters",
        lazy_linear.parameters().len()
    );
    println!(
        "   - Inferred input features: {:?}",
        lazy_linear.in_features()
    );
    assert!(lazy_linear.is_initialized());

    // Now we can perform forward pass
    let output = lazy_linear.forward(&input)?;
    println!("   - Output shape: {:?}", output.shape().dims());
    println!("   ✓ LazyLinear working correctly\n");

    Ok(())
}

fn lazy_conv1d_example() -> Result<()> {
    println!("2. LazyConv1d Example:");
    println!("   - Creates a 1D convolution layer without specifying input channels");
    println!("   - Automatically infers input_channels from first forward pass");

    // Create a lazy conv1d layer: 32 output channels, kernel size 3
    let mut lazy_conv = LazyConv1d::simple(32, 3, true);

    println!(
        "   - Before initialization: {} parameters",
        lazy_conv.parameters().len()
    );
    assert!(!lazy_conv.is_initialized());

    // Create input tensor [batch, channels, length]
    let input = randn::<f32>(&[8, 16, 100])?; // 16 input channels, sequence length 100

    // Initialize the layer
    lazy_conv.initialize_lazy(&input)?;

    println!(
        "   - After initialization: {} parameters",
        lazy_conv.parameters().len()
    );
    println!(
        "   - Inferred input channels: {:?}",
        lazy_conv.in_channels()
    );
    assert!(lazy_conv.is_initialized());

    // Forward pass
    let output = lazy_conv.forward(&input)?;
    println!("   - Output shape: {:?}", output.shape().dims());
    println!("   ✓ LazyConv1d working correctly\n");

    Ok(())
}

fn lazy_conv2d_example() -> Result<()> {
    println!("3. LazyConv2d Example:");
    println!("   - Creates a 2D convolution layer without specifying input channels");
    println!("   - Automatically infers input_channels from first forward pass");

    // Create a lazy conv2d layer: 64 output channels, 3x3 kernel
    let mut lazy_conv = LazyConv2d::simple(64, (3, 3), false);

    println!(
        "   - Before initialization: {} parameters",
        lazy_conv.parameters().len()
    );
    assert!(!lazy_conv.is_initialized());

    // Create input tensor [batch, channels, height, width]
    let input = randn::<f32>(&[4, 3, 224, 224])?; // RGB image input

    // Initialize the layer
    lazy_conv.initialize_lazy(&input)?;

    println!(
        "   - After initialization: {} parameters",
        lazy_conv.parameters().len()
    );
    println!(
        "   - Inferred input channels: {:?}",
        lazy_conv.in_channels()
    );
    assert!(lazy_conv.is_initialized());

    // Forward pass
    let output = lazy_conv.forward(&input)?;
    println!("   - Output shape: {:?}", output.shape().dims());
    println!("   ✓ LazyConv2d working correctly\n");

    Ok(())
}

fn error_handling_example() -> Result<()> {
    println!("4. Error Handling Example:");
    println!("   - Demonstrates what happens when forgetting to initialize lazy modules");

    let lazy_linear = LazyLinear::new(10, true);
    let input = randn::<f32>(&[32, 20])?;

    // Try to use the layer without initialization
    match lazy_linear.forward(&input) {
        Ok(_) => println!("   - Unexpected success!"),
        Err(e) => {
            println!("   - Expected error: {}", e);
            println!("   - This error message guides users to call initialize_lazy() first");
        }
    }

    println!("   ✓ Error handling working correctly\n");

    Ok(())
}

fn sequential_lazy_example() -> Result<()> {
    println!("5. Sequential Network with Lazy Modules:");
    println!("   - Shows how to use lazy modules in a sequential network");

    // Create a network with lazy modules
    // Note: In a real implementation, you'd use a proper container or trait objects

    // First layer: LazyLinear that will infer input size
    let mut lazy_linear1 = LazyLinear::new(128, true);

    // Second layer: LazyLinear that will infer input size from first layer output
    let mut lazy_linear2 = LazyLinear::new(64, true);

    // Third layer: LazyLinear for final output
    let mut lazy_linear3 = LazyLinear::new(10, true);

    // Create input data
    let input = randn::<f32>(&[32, 256])?; // 32 samples, 256 features

    // Initialize and forward pass through the network
    println!("   - Initializing network layer by layer...");

    // Layer 1: Input -> 128
    lazy_linear1.initialize_lazy(&input)?;
    let output1 = lazy_linear1.forward(&input)?;
    println!("   - Layer 1 output shape: {:?}", output1.shape().dims());

    // Layer 2: 128 -> 64
    lazy_linear2.initialize_lazy(&output1)?;
    let output2 = lazy_linear2.forward(&output1)?;
    println!("   - Layer 2 output shape: {:?}", output2.shape().dims());

    // Layer 3: 64 -> 10
    lazy_linear3.initialize_lazy(&output2)?;
    let final_output = lazy_linear3.forward(&output2)?;
    println!("   - Final output shape: {:?}", final_output.shape().dims());

    println!("   ✓ Sequential lazy network working correctly\n");

    Ok(())
}
