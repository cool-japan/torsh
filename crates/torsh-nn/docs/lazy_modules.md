# Lazy Module Usage Guide

## Overview

Lazy modules in torsh-nn are neural network layers that automatically infer their input dimensions from the first forward pass. This is particularly useful when you don't know the exact input dimensions at model creation time, similar to PyTorch's lazy modules.

## Key Concepts

### 1. Lazy Initialization
- Lazy modules are created without specifying input dimensions
- They infer input dimensions from the first tensor passed to them
- Parameters are only created after initialization

### 2. Two-Phase Usage Pattern
1. **Creation Phase**: Create the lazy module with output dimensions only
2. **Initialization Phase**: Call `initialize_lazy()` with a sample input tensor

## Available Lazy Modules

### LazyLinear
- **Purpose**: Fully connected layer that infers input features
- **Usage**: `LazyLinear::new(out_features, bias)`
- **Input**: Any tensor where the last dimension is the feature dimension
- **Initialization**: Automatically infers `in_features` from input tensor

```rust
use torsh_nn::layers::lazy::{LazyLinear, LazyModule};
use torsh_tensor::creation::randn;

let mut layer = LazyLinear::new(128, true);
let input = randn::<f32>(&[32, 64])?; // [batch, features]
layer.initialize_lazy(&input)?;
let output = layer.forward(&input)?; // [32, 128]
```

### LazyConv1d
- **Purpose**: 1D convolution that infers input channels
- **Usage**: `LazyConv1d::new(out_channels, kernel_size, stride, padding, dilation, groups, bias)`
- **Simple Usage**: `LazyConv1d::simple(out_channels, kernel_size, bias)`
- **Input**: Tensor with shape `[batch, channels, length]`

```rust
use torsh_nn::layers::lazy::{LazyConv1d, LazyModule};
use torsh_tensor::creation::randn;

let mut layer = LazyConv1d::simple(32, 3, true);
let input = randn::<f32>(&[8, 16, 100])?; // [batch, channels, length]
layer.initialize_lazy(&input)?;
let output = layer.forward(&input)?;
```

### LazyConv2d
- **Purpose**: 2D convolution that infers input channels
- **Usage**: `LazyConv2d::new(out_channels, kernel_size, stride, padding, dilation, groups, bias)`
- **Simple Usage**: `LazyConv2d::simple(out_channels, kernel_size, bias)`
- **Input**: Tensor with shape `[batch, channels, height, width]`

```rust
use torsh_nn::layers::lazy::{LazyConv2d, LazyModule};
use torsh_tensor::creation::randn;

let mut layer = LazyConv2d::simple(64, (3, 3), false);
let input = randn::<f32>(&[4, 3, 224, 224])?; // [batch, channels, height, width]
layer.initialize_lazy(&input)?;
let output = layer.forward(&input)?;
```

## Usage Patterns

### 1. Basic Usage Pattern
```rust
// 1. Create lazy module
let mut lazy_layer = LazyLinear::new(output_size, true);

// 2. Initialize with sample input
let sample_input = randn::<f32>(&[batch_size, input_size])?;
lazy_layer.initialize_lazy(&sample_input)?;

// 3. Use normally
let output = lazy_layer.forward(&input)?;
```

### 2. Sequential Network Pattern
```rust
// Create multiple lazy layers
let mut layer1 = LazyLinear::new(128, true);
let mut layer2 = LazyLinear::new(64, true);
let mut layer3 = LazyLinear::new(10, true);

// Initialize chain by chain
let input = randn::<f32>(&[32, 256])?;

layer1.initialize_lazy(&input)?;
let out1 = layer1.forward(&input)?;

layer2.initialize_lazy(&out1)?;
let out2 = layer2.forward(&out1)?;

layer3.initialize_lazy(&out2)?;
let final_output = layer3.forward(&out2)?;
```

### 3. Error Handling Pattern
```rust
let lazy_layer = LazyLinear::new(10, true);
let input = randn::<f32>(&[32, 20])?;

// This will return an error with helpful message
match lazy_layer.forward(&input) {
    Ok(_) => println!("Success!"),
    Err(e) => {
        // Error message will be:
        // "LazyLinear not initialized. Detected in_features=20. Call initialize_lazy(20) first."
        println!("Error: {}", e);
    }
}
```

## LazyModule Trait

The `LazyModule` trait provides a unified interface for lazy initialization:

```rust
pub trait LazyModule: Module {
    fn initialize_lazy(&mut self, input: &Tensor) -> Result<()>;
}
```

This trait is implemented by all lazy modules and allows for generic lazy initialization.

## Benefits

### 1. Flexibility
- No need to specify input dimensions at creation time
- Useful for dynamic architectures or when input shape is unknown

### 2. Automatic Dimension Inference
- Automatically calculates required parameter shapes
- Reduces boilerplate code and potential errors

### 3. Clear Error Messages
- Provides helpful error messages when initialization is forgotten
- Guides users to the correct usage pattern

## Best Practices

### 1. Always Initialize Before Use
```rust
// ❌ Wrong - will cause runtime error
let layer = LazyLinear::new(10, true);
let output = layer.forward(&input)?; // Error!

// ✅ Correct - initialize first
let mut layer = LazyLinear::new(10, true);
layer.initialize_lazy(&input)?;
let output = layer.forward(&input)?; // Success!
```

### 2. Check Initialization State
```rust
let layer = LazyLinear::new(10, true);
if !layer.is_initialized() {
    layer.initialize_lazy(&input)?;
}
```

### 3. Use Simple Constructors When Possible
```rust
// ❌ Verbose
let layer = LazyConv2d::new(64, (3, 3), (1, 1), (0, 0), (1, 1), 1, true);

// ✅ Concise
let layer = LazyConv2d::simple(64, (3, 3), true);
```

## Common Pitfalls

### 1. Forgetting to Initialize
- **Problem**: Calling `forward()` without `initialize_lazy()`
- **Solution**: Always call `initialize_lazy()` before first use
- **Detection**: Runtime error with helpful message

### 2. Multiple Initialization Attempts
- **Problem**: Calling `initialize_lazy()` multiple times
- **Solution**: Check `is_initialized()` first, or just call it (it's idempotent)
- **Detection**: No error, but unnecessary work

### 3. Wrong Input Dimensions
- **Problem**: Passing input with wrong number of dimensions
- **Solution**: Ensure input tensor has correct number of dimensions for the layer type
- **Detection**: Runtime error with dimension requirements

## Advanced Usage

### 1. Custom Initialization Logic
```rust
let mut layer = LazyLinear::new(128, true);
let input = randn::<f32>(&[32, 64])?;

// Initialize if needed
if !layer.is_initialized() {
    layer.initialize_lazy(&input)?;
    
    // Optional: Access inferred dimensions
    println!("Inferred input features: {:?}", layer.in_features());
}
```

### 2. Conditional Layer Creation
```rust
fn create_adaptive_network(input_sample: &Tensor) -> Result<Vec<Box<dyn Module>>> {
    let mut layers = Vec::new();
    
    // First layer adapts to input
    let mut layer1 = LazyLinear::new(128, true);
    layer1.initialize_lazy(input_sample)?;
    layers.push(Box::new(layer1) as Box<dyn Module>);
    
    // Additional layers...
    Ok(layers)
}
```

## Migration from Regular Modules

### Before (Regular Modules)
```rust
let linear = Linear::new(input_features, output_features, true)?;
let conv2d = Conv2d::new(in_channels, out_channels, (3, 3), (1, 1), (1, 1), 1, true)?;
```

### After (Lazy Modules)
```rust
let mut linear = LazyLinear::new(output_features, true);
linear.initialize_lazy(&input)?;

let mut conv2d = LazyConv2d::simple(out_channels, (3, 3), true);
conv2d.initialize_lazy(&input)?;
```

## Performance Considerations

- **Initialization Cost**: One-time cost during first forward pass
- **Runtime Cost**: No additional cost after initialization
- **Memory Usage**: Same as regular modules after initialization
- **Thread Safety**: Lazy modules use mutex for thread-safe initialization

## Examples

See `examples/lazy_module_usage.rs` for comprehensive examples demonstrating all lazy module types and usage patterns.