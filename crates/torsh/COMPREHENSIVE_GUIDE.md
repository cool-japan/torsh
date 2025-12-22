# ToRSh Framework: Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Concepts](#core-concepts)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Advanced Features](#advanced-features)
10. [Migration Guide](#migration-guide)
11. [Examples](#examples)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)

## Introduction

ToRSh (Tensor Operations in Rust with Sharding) is a production-ready deep learning framework built entirely in Rust. It provides a PyTorch-compatible API with superior performance, memory safety, and zero-cost abstractions. ToRSh leverages Rust's type system and the scirs2 ecosystem to deliver high-performance tensor operations, automatic differentiation, and comprehensive neural network capabilities.

### Key Features

- **Memory Safety**: Rust's ownership model eliminates memory leaks and segmentation faults
- **Zero-Cost Abstractions**: High-level API with no runtime overhead
- **PyTorch Compatibility**: Familiar API for easy migration from PyTorch
- **Multi-Backend Support**: CPU, CUDA, Metal, and WebGPU backends
- **Automatic Differentiation**: Complete autograd system with gradient computation
- **Production Ready**: Comprehensive error handling and robust implementations
- **SIMD Optimizations**: AVX-512, AVX2, and ARM NEON vectorization
- **Distributed Training**: Multi-GPU and multi-node training support

## Architecture Overview

ToRSh uses a modular workspace architecture with specialized crates:

### Core Infrastructure
- **torsh-core**: Core types (Device, DType, Shape, Storage)
- **torsh-tensor**: Tensor implementation with strided storage
- **torsh-autograd**: Automatic differentiation engine
- **torsh-backend**: Backend abstraction layer

### Backend Implementations
- **torsh-backend-cpu**: CPU backend with SIMD optimizations
- **torsh-backend-cuda**: CUDA GPU backend
- **torsh-backend-metal**: Apple Metal backend
- **torsh-backend-webgpu**: WebGPU backend

### Neural Networks
- **torsh-nn**: Neural network modules and layers
- **torsh-optim**: Optimization algorithms
- **torsh-functional**: Functional operations API

### Specialized Modules
- **torsh-data**: Data loading and preprocessing
- **torsh-vision**: Computer vision operations
- **torsh-text**: Natural language processing
- **torsh-sparse**: Sparse tensor operations
- **torsh-linalg**: Linear algebra operations
- **torsh-special**: Special mathematical functions

### Advanced Features
- **torsh-distributed**: Distributed training
- **torsh-quantization**: Model quantization
- **torsh-profiler**: Performance profiling
- **torsh-jit**: Just-in-time compilation
- **torsh-fx**: Graph optimization

## Installation

### Prerequisites
- Rust 1.70+ (latest stable recommended)
- CUDA 11.8+ (for GPU support)
- OpenMP (for CPU parallelization)

### Basic Installation
```bash
# Add to Cargo.toml
[dependencies]
torsh = "0.1.0-alpha.2"
```

### Full Installation with All Features
```bash
# Add to Cargo.toml
[dependencies]
torsh = { version = "0.1.0-alpha.2", features = ["full"] }
```

### Feature Flags
- `cuda`: CUDA GPU support
- `metal`: Apple Metal support
- `webgpu`: WebGPU support
- `distributed`: Distributed training
- `quantization`: Model quantization
- `profiler`: Performance profiling
- `full`: All features enabled

## Quick Start

### Basic Tensor Operations
```rust
use torsh::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors
    let a = Tensor::randn(&[2, 3], DType::F32, Device::Cpu)?;
    let b = Tensor::ones(&[2, 3], DType::F32, Device::Cpu)?;
    
    // Basic operations
    let c = a.add(&b)?;
    let d = c.mul(&Tensor::scalar(2.0, DType::F32, Device::Cpu)?)?;
    
    println!("Result: {:?}", d);
    Ok(())
}
```

### Neural Network Training
```rust
use torsh::prelude::*;
use torsh::nn::{Linear, Module};
use torsh::optim::Adam;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available();
    
    // Create model
    let model = Linear::new(784, 10, true, device)?;
    
    // Create optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.001)?;
    
    // Training loop
    for epoch in 0..100 {
        // Forward pass
        let output = model.forward(&input)?;
        let loss = F::cross_entropy(&output, &targets)?;
        
        // Backward pass
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;
        
        if epoch % 10 == 0 {
            println!("Epoch {}, Loss: {:.4}", epoch, loss.item::<f32>()?);
        }
    }
    
    Ok(())
}
```

## Core Concepts

### Tensors
Tensors are the fundamental data structure in ToRSh. They represent multi-dimensional arrays with automatic differentiation capabilities.

```rust
// Creation
let t1 = Tensor::zeros(&[2, 3], DType::F32, Device::Cpu)?;
let t2 = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], DType::F32, Device::Cpu)?;
let t3 = Tensor::randn(&[10, 20], DType::F32, Device::Cpu)?;

// Operations
let result = t1.add(&t2)?.mul(&t3)?;
```

### Automatic Differentiation
ToRSh provides automatic differentiation through the autograd system:

```rust
let x = Tensor::randn(&[1, 1], DType::F32, Device::Cpu)?;
x.set_requires_grad(true);

let y = x.pow(&Tensor::scalar(2.0, DType::F32, Device::Cpu)?)?;
y.backward()?;

let grad = x.grad().unwrap();
println!("Gradient: {:?}", grad);
```

### Devices
ToRSh supports multiple compute backends:

```rust
// CPU
let cpu_tensor = Tensor::randn(&[2, 3], DType::F32, Device::Cpu)?;

// CUDA
let cuda_tensor = Tensor::randn(&[2, 3], DType::F32, Device::Cuda(0))?;

// Metal
let metal_tensor = Tensor::randn(&[2, 3], DType::F32, Device::Metal)?;

// Automatic device selection
let device = Device::cuda_if_available();
```

### Data Types
ToRSh supports various data types:

```rust
// Floating point
let f32_tensor = Tensor::zeros(&[2, 3], DType::F32, Device::Cpu)?;
let f64_tensor = Tensor::zeros(&[2, 3], DType::F64, Device::Cpu)?;

// Integers
let i32_tensor = Tensor::zeros(&[2, 3], DType::I32, Device::Cpu)?;
let i64_tensor = Tensor::zeros(&[2, 3], DType::I64, Device::Cpu)?;

// Boolean
let bool_tensor = Tensor::zeros(&[2, 3], DType::Bool, Device::Cpu)?;
```

## API Reference

### Tensor Creation
```rust
// Zeros and ones
Tensor::zeros(&[2, 3], DType::F32, Device::Cpu)?;
Tensor::ones(&[2, 3], DType::F32, Device::Cpu)?;

// Random tensors
Tensor::randn(&[2, 3], DType::F32, Device::Cpu)?;
Tensor::rand(&[2, 3], DType::F32, Device::Cpu)?;

// From data
Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], DType::F32, Device::Cpu)?;

// Ranges
Tensor::arange(0.0, 10.0, 1.0, DType::F32, Device::Cpu)?;
```

### Tensor Operations
```rust
// Arithmetic
let result = a.add(&b)?;
let result = a.sub(&b)?;
let result = a.mul(&b)?;
let result = a.div(&b)?;

// Matrix operations
let result = a.matmul(&b)?;
let result = a.transpose(0, 1)?;

// Reductions
let sum = a.sum(Some(&[0]), false)?;
let mean = a.mean(Some(&[1]), true)?;
let max = a.max(Some(&[0]), false)?;
```

### Neural Network Layers
```rust
// Linear layer
let linear = Linear::new(784, 128, true, device)?;

// Convolutional layer
let conv = Conv2d::new(3, 64, 3, 1, 1, true, device)?;

// Batch normalization
let bn = BatchNorm2d::new(64, device)?;

// Activation functions
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
```

### Optimizers
```rust
// SGD
let sgd = SGD::new(model.parameters(), 0.01)?;

// Adam
let adam = Adam::new(model.parameters(), 0.001)?;

// AdamW
let adamw = AdamW::new(model.parameters(), 0.001)?;
```

## Best Practices

### Memory Management
- Use `Device::cuda_if_available()` for automatic device selection
- Batch operations to reduce memory allocations
- Use in-place operations when possible (`add_` instead of `add`)
- Clear gradients with `optimizer.zero_grad()` before backward pass

### Performance
- Use appropriate data types (`F32` for most neural networks)
- Leverage SIMD operations for CPU computations
- Use mixed precision training for memory efficiency
- Profile your code with `torsh-profiler`

### Error Handling
- Always handle `Result` types properly
- Use `?` operator for error propagation
- Provide meaningful error messages
- Validate tensor shapes before operations

## Performance Optimization

### SIMD Optimizations
ToRSh automatically uses SIMD instructions when available:

```rust
// Automatically vectorized operations
let result = a.add(&b)?; // Uses AVX-512 on supported CPUs
```

### GPU Acceleration
```rust
// Move tensors to GPU
let gpu_tensor = cpu_tensor.to_device(Device::Cuda(0))?;

// All operations run on GPU
let result = gpu_tensor.matmul(&other_gpu_tensor)?;
```

### Memory Optimization
```rust
// Use in-place operations
a.add_(&b)?; // Modifies 'a' directly

// Gradient checkpointing
model.set_gradient_checkpointing(true);
```

## Advanced Features

### Distributed Training
```rust
use torsh::distributed::DistributedDataParallel;

let model = DistributedDataParallel::new(model, &[0, 1])?;
```

### Model Quantization
```rust
use torsh::quantization::quantize_model;

let quantized_model = quantize_model(model, QuantizationConfig::default())?;
```

### JIT Compilation
```rust
use torsh::jit::trace;

let traced_model = trace(model, &example_input)?;
```

## Migration Guide

### From PyTorch
ToRSh provides a PyTorch-compatible API:

```python
# PyTorch
import torch
x = torch.randn(2, 3)
y = x.add(1.0)
```

```rust
// ToRSh
use torsh::prelude::*;
let x = Tensor::randn(&[2, 3], DType::F32, Device::Cpu)?;
let y = x.add(&Tensor::scalar(1.0, DType::F32, Device::Cpu)?)?;
```

### Key Differences
- Explicit error handling with `Result` types
- Explicit device and dtype specification
- Memory safety guarantees
- Zero-cost abstractions

## Examples

### Image Classification
```rust
use torsh::prelude::*;
use torsh::vision::models::ResNet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = ResNet::resnet18(1000, Device::cuda_if_available())?;
    
    // Load and preprocess image
    let image = load_image("image.jpg")?;
    let preprocessed = preprocess_image(image)?;
    
    // Forward pass
    let output = model.forward(&preprocessed)?;
    let probabilities = F::softmax(&output, 1)?;
    
    println!("Predictions: {:?}", probabilities);
    Ok(())
}
```

### Natural Language Processing
```rust
use torsh::prelude::*;
use torsh::text::models::BERT;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = BERT::from_pretrained("bert-base-uncased")?;
    
    // Tokenize text
    let tokens = tokenize("Hello, world!")?;
    let input_ids = Tensor::from_slice(&tokens, &[1, tokens.len()], DType::I64, Device::Cpu)?;
    
    // Forward pass
    let output = model.forward(&input_ids)?;
    
    println!("Embeddings: {:?}", output);
    Ok(())
}
```

## Troubleshooting

### Common Issues

#### Compilation Errors
- Ensure you're using Rust 1.70+
- Check feature flags are correctly specified
- Verify CUDA installation if using GPU features

#### Runtime Errors
- Check tensor shapes match for operations
- Verify tensors are on the same device
- Ensure sufficient memory is available

#### Performance Issues
- Use appropriate batch sizes
- Enable SIMD optimizations
- Consider using mixed precision
- Profile with `torsh-profiler`

### Memory Issues
```rust
// Monitor memory usage
let memory_stats = Device::memory_stats(Device::Cuda(0))?;
println!("Memory usage: {} MB", memory_stats.used / 1024 / 1024);

// Clear cache
Device::empty_cache(Device::Cuda(0))?;
```

### Debug Tips
```rust
// Enable debug mode
std::env::set_var("TORSH_DEBUG", "1");

// Check tensor properties
println!("Shape: {:?}", tensor.shape());
println!("Device: {:?}", tensor.device());
println!("DType: {:?}", tensor.dtype());
```

## Contributing

### Development Setup
```bash
git clone https://github.com/your-repo/torsh.git
cd torsh
cargo build --workspace
cargo test --workspace
```

### Testing
```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test --package torsh-tensor

# Run with specific features
cargo test --workspace --features "cuda"
```

### Code Style
- Follow Rust idioms and conventions
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Write comprehensive tests

### Documentation
- Document all public APIs
- Include examples in documentation
- Keep README files updated
- Write clear error messages

This comprehensive guide provides everything you need to get started with ToRSh and use it effectively for your deep learning projects. The framework combines the performance and safety of Rust with the familiar PyTorch API, making it an excellent choice for production machine learning applications.