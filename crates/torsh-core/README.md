# torsh-core

Core types and traits for the ToRSh deep learning framework.

## Overview

This crate provides the fundamental building blocks used throughout ToRSh:

- **Device abstraction**: Unified interface for CPU, CUDA, Metal, and WebGPU backends
- **Data types**: Support for various tensor element types (f32, f64, i32, etc.)
- **Shape utilities**: Shape manipulation, broadcasting, and stride calculations
- **Storage abstraction**: Backend-agnostic tensor storage with reference counting
- **Error types**: Comprehensive error handling for ToRSh operations

## Features

- `std` (default): Standard library support
- `no_std`: No standard library (for embedded targets)
- `serialize`: Serialization support via serde

## Usage

```rust
use torsh_core::prelude::*;

// Create a shape
let shape = Shape::new(vec![2, 3, 4]);
println!("Shape: {}, elements: {}", shape, shape.numel());

// Device management
let device = CpuDevice;
println!("Device: {}", device.name());

// Data types
let dtype = DType::F32;
println!("Data type: {}, size: {} bytes", dtype, dtype.size());
```

## Integration with SciRS2

This crate builds on top of [scirs2](https://crates.io/crates/scirs2) for core scientific computing functionality, providing a PyTorch-compatible API layer.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.