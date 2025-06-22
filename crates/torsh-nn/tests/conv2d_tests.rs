//! Tests for Conv2d layer

use torsh_nn::modules::Conv2d;
use torsh_nn::Module;
use torsh_tensor::creation::*;
use torsh_core::error::Result;

#[test]
fn test_conv2d_basic() -> Result<()> {
    // Create a simple Conv2d layer
    let conv = Conv2d::new(
        3,  // in_channels
        16, // out_channels
        (3, 3), // kernel_size
        None, // stride (default: 1)
        None, // padding (default: 0)
        None, // dilation (default: 1)
        None, // groups (default: 1)
        true, // bias
    );
    
    // Create input tensor [batch_size=2, channels=3, height=32, width=32]
    let input = randn(&[2, 3, 32, 32]);
    
    // Forward pass
    let output = conv.forward(&input)?;
    
    // Check output shape
    // With kernel_size=3x3, stride=1, padding=0:
    // output_height = (32 - 3 + 1) = 30
    // output_width = (32 - 3 + 1) = 30
    let output_shape_obj = output.shape();
    let output_shape = output_shape_obj.dims();
    assert_eq!(output_shape, &[2, 16, 30, 30]);
    
    Ok(())
}

#[test]
fn test_conv2d_with_padding() -> Result<()> {
    // Create Conv2d with padding
    let conv = Conv2d::new(
        3,  // in_channels
        8,  // out_channels
        (5, 5), // kernel_size
        None, // stride (default: 1)
        Some((2, 2)), // padding = 2
        None, // dilation (default: 1)
        None, // groups (default: 1)
        false, // no bias
    );
    
    // Create input tensor
    let input = ones(&[1, 3, 10, 10]);
    
    // Forward pass
    let output = conv.forward(&input)?;
    
    // Check output shape
    // With kernel_size=5x5, stride=1, padding=2:
    // output_height = (10 + 2*2 - 5 + 1) = 10 (same as input)
    // output_width = (10 + 2*2 - 5 + 1) = 10 (same as input)
    let output_shape_obj = output.shape();
    let output_shape = output_shape_obj.dims();
    assert_eq!(output_shape, &[1, 8, 10, 10]);
    
    Ok(())
}

#[test]
fn test_conv2d_with_stride() -> Result<()> {
    // Create Conv2d with stride
    let conv = Conv2d::new(
        1,  // in_channels
        4,  // out_channels
        (3, 3), // kernel_size
        Some((2, 2)), // stride = 2
        None, // padding (default: 0)
        None, // dilation (default: 1)
        None, // groups (default: 1)
        true, // bias
    );
    
    // Create input tensor
    let input = zeros(&[1, 1, 8, 8]);
    
    // Forward pass
    let output = conv.forward(&input)?;
    
    // Check output shape
    // With kernel_size=3x3, stride=2, padding=0:
    // output_height = (8 - 3) / 2 + 1 = 3
    // output_width = (8 - 3) / 2 + 1 = 3
    let output_shape_obj = output.shape();
    let output_shape = output_shape_obj.dims();
    assert_eq!(output_shape, &[1, 4, 3, 3]);
    
    Ok(())
}

#[test]
fn test_conv2d_depthwise() -> Result<()> {
    // Create depthwise Conv2d (groups = in_channels)
    let conv = Conv2d::new(
        6,  // in_channels
        6,  // out_channels (same as in_channels for depthwise)
        (3, 3), // kernel_size
        None, // stride (default: 1)
        None, // padding (default: 0)
        None, // dilation (default: 1)
        Some(6), // groups = in_channels (depthwise)
        true, // bias
    );
    
    // Create input tensor
    let input = randn(&[2, 6, 16, 16]);
    
    // Forward pass
    let output = conv.forward(&input)?;
    
    // Check output shape
    let output_shape_obj = output.shape();
    let output_shape = output_shape_obj.dims();
    assert_eq!(output_shape, &[2, 6, 14, 14]);
    
    Ok(())
}

#[test]
fn test_conv2d_parameters() {
    // Create Conv2d layer
    let conv = Conv2d::new(
        3,  // in_channels
        16, // out_channels
        (5, 5), // kernel_size
        None, None, None, None,
        true, // bias
    );
    
    // Check parameters
    let params = conv.named_parameters();
    assert!(params.contains_key("weight"));
    assert!(params.contains_key("bias"));
    
    // Check weight shape
    let weight_shape_obj = params["weight"].read();
    let weight_shape = weight_shape_obj.shape().dims().to_vec();
    assert_eq!(weight_shape, vec![16, 3, 5, 5]); // [out_channels, in_channels, kH, kW]
    
    // Check bias shape
    let bias_shape_obj = params["bias"].read();
    let bias_shape = bias_shape_obj.shape().dims().to_vec();
    assert_eq!(bias_shape, vec![16]); // [out_channels]
}

#[test]
fn test_conv2d_no_bias() {
    // Create Conv2d without bias
    let conv = Conv2d::new(
        3,  // in_channels
        8,  // out_channels
        (3, 3), // kernel_size
        None, None, None, None,
        false, // no bias
    );
    
    // Check parameters
    let params = conv.named_parameters();
    assert!(params.contains_key("weight"));
    assert!(!params.contains_key("bias")); // Should not have bias
    assert_eq!(params.len(), 1);
}

#[test]
fn test_conv2d_invalid_input_shape() {
    let conv = Conv2d::new(3, 16, (3, 3), None, None, None, None, true);
    
    // Test with 3D input (should fail)
    let input_3d = randn(&[2, 3, 32]); // Missing width dimension
    assert!(conv.forward(&input_3d).is_err());
    
    // Test with 2D input (should fail)
    let input_2d = randn(&[3, 32]); // Missing batch and width dimensions
    assert!(conv.forward(&input_2d).is_err());
}

#[test]
fn test_conv2d_training_mode() {
    let mut conv = Conv2d::new(3, 16, (3, 3), None, None, None, None, true);
    
    // Check initial training mode (should be true)
    assert!(conv.training());
    
    // Set to eval mode
    conv.train(false);
    assert!(!conv.training());
    
    // Set back to training mode
    conv.train(true);
    assert!(conv.training());
}