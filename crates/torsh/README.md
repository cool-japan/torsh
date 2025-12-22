# torsh

The main crate for ToRSh - A blazingly fast, production-ready deep learning framework written in pure Rust.

## Overview

This is the primary entry point for the ToRSh framework, providing convenient access to all functionality through a unified API.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
torsh = "0.1.0-alpha.2"
```

## Quick Start

```rust
use torsh::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors
    let x = tensor![[1.0, 2.0], [3.0, 4.0]];
    let y = tensor![[5.0, 6.0], [7.0, 8.0]];
    
    // Matrix multiplication
    let z = x.matmul(&y)?;
    println!("Result: {:?}", z);
    
    // Automatic differentiation
    let a = tensor![2.0].requires_grad_(true);
    let b = a.pow(2.0)? + a * 3.0;
    b.backward()?;
    println!("Gradient: {:?}", a.grad()?);
    
    Ok(())
}
```

## Available Features

- `default`: Includes `std`, `nn`, `optim`, and `data`
- `std`: Standard library support (enabled by default)
- `nn`: Neural network modules
- `optim`: Optimization algorithms
- `data`: Data loading utilities
- `cuda`: CUDA backend support
- `wgpu`: WebGPU backend support
- `metal`: Metal backend support (Apple Silicon)
- `serialize`: Serialization support
- `full`: All features

## Module Structure

The crate re-exports functionality from specialized sub-crates:

- **Core** (`torsh::core`): Basic types and traits
- **Tensor** (`torsh::tensor`): Tensor operations
- **Autograd** (`torsh::autograd`): Automatic differentiation
- **NN** (`torsh::nn`): Neural network layers
- **Optim** (`torsh::optim`): Optimizers
- **Data** (`torsh::data`): Data loading

## F Namespace

Similar to PyTorch's `torch.nn.functional`, ToRSh provides functional operations in the `F` namespace:

```rust
use torsh::F;

let output = F::relu(&input);
let output = F::softmax(&logits, -1)?;
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.