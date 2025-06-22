# ToRSh - Tensor Operations in Rust with Sharding

<div align="center">

[![Crates.io](https://img.shields.io/crates/v/torsh.svg)](https://crates.io/crates/torsh)
[![Documentation](https://docs.rs/torsh/badge.svg)](https://docs.rs/torsh)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](./LICENSE-MIT)
[![Build Status](https://github.com/cool-japan/torsh/workflows/CI/badge.svg)](https://github.com/cool-japan/torsh/actions)

**A blazingly fast, production-ready deep learning framework written in pure Rust**

[Documentation](https://docs.rs/torsh) | [Examples](./examples) | [Benchmarks](./benches) | [Contributing](./CONTRIBUTING.md)

</div>

## Overview

ToRSh (Tensor Operations in Rust with Sharding) is a next-generation deep learning framework built with Rust's zero-cost abstractions, memory safety, and fearless concurrency. Currently in alpha development, ToRSh features a complete tensor system, automatic differentiation, neural network modules, and comprehensive test coverage with 123+ passing tests.

### Key Features

- 🚀 **High Performance**: Designed for speed with SIMD vectorization and optimized memory management
- 🛡️ **Memory Safety**: Compile-time guarantees eliminate entire classes of bugs common in C++ ML frameworks
- 🌍 **Portability**: Cross-platform design with modular backend architecture
- ✅ **Type Safety**: Rust's type system prevents runtime errors and dimension mismatches
- 🔧 **Modern Architecture**: Clean backend abstraction and fearless concurrency
- 🦀 **Pure Rust**: Built entirely in Rust, leveraging the ecosystem and tooling

## Quick Start

Add ToRSh to your `Cargo.toml`:

```toml
[dependencies]
torsh = "0.1.0"
```

### Basic Example

```rust
use torsh::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors
    let x = tensor![[1.0, 2.0], [3.0, 4.0]];
    let y = tensor![[5.0, 6.0], [7.0, 8.0]];
    
    // Basic operations
    let z = x.matmul(&y)?;
    println!("Result: {:?}", z);
    
    // Automatic differentiation
    let a = Tensor::new(&[2.0]).requires_grad(true);
    let b = a.pow(2.0) + a * 3.0;
    b.backward()?;
    println!("Gradient: {:?}", a.grad()?); // 2*a + 3 = 7.0
    
    Ok(())
}
```

### Neural Network Example

```rust
use torsh::prelude::*;
use torsh::nn::{Module, Linear, Sequential};
use torsh::optim::{Adam, Optimizer};

// Define a simple neural network
struct SimpleNet {
    layers: Sequential,
}

impl SimpleNet {
    fn new() -> Self {
        Self {
            layers: Sequential::new()
                .add(Linear::<784, 128>::new())
                .add_fn(|x| x.relu())
                .add(Linear::<128, 10>::new()),
        }
    }
}

impl Module for SimpleNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
        self.layers.forward(input)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create model and optimizer
    let mut model = SimpleNet::new();
    let mut optimizer = Adam::new(model.parameters(), 0.001);
    
    // Training loop
    for epoch in 0..10 {
        let input = Tensor::randn(&[32, 784]); // Batch of 32 samples
        let target = Tensor::randint(0, 10, &[32])?;
        
        // Forward pass
        let output = model.forward(&input)?;
        let loss = output.cross_entropy(&target)?;
        
        // Backward pass
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;
        
        println!("Epoch {}: Loss = {}", epoch, loss.item());
    }
    
    Ok(())
}
```

## Performance

ToRSh achieves superior performance through:

- **Zero-cost abstractions**: Compile-time optimizations with no runtime overhead
- **SIMD vectorization**: Automatic use of CPU vector instructions
- **Kernel fusion**: JIT compilation combines operations for fewer memory accesses
- **Efficient memory management**: Custom allocators and tensor views minimize copying
- **Parallelism**: Leverages Rust's fearless concurrency for multi-threaded operations

### Benchmarks

Benchmarking infrastructure is in place with the `torsh-benches` crate. Performance benchmarks against PyTorch will be conducted as the framework matures. Current focus is on correctness and API completeness in the alpha release.

## Architecture

ToRSh is built as a modular workspace with specialized crates:

- `torsh-core`: Core types and traits
- `torsh-tensor`: Tensor implementation with strided storage
- `torsh-autograd`: Automatic differentiation engine
- `torsh-nn`: Neural network modules and layers
- `torsh-optim`: Optimization algorithms
- `torsh-backends`: Backend trait definitions
- Various backend implementations (CPU, CUDA, WebGPU, Metal)

## Supported Backends

- ✅ **CPU**: Optimized with SIMD instructions and multi-threading
- 🚧 **CUDA**: NVIDIA GPU support (in progress)
- 🚧 **WebGPU**: Cross-platform GPU (planned)
- 🚧 **Metal**: Apple Silicon GPU (planned)
- 🚧 **OpenCL**: Broad GPU support (planned)

## Installation

### Default Installation (CPU only)

```bash
cargo add torsh
```

### With CUDA Support

```bash
cargo add torsh --features cuda
```

### With All Features

```bash
cargo add torsh --features full
```

## Documentation

- [API Documentation](https://docs.rs/torsh)
- [User Guide](./docs/guide)
- [Architecture Overview](./docs/architecture)
- [Migration from PyTorch](./docs/migration)
- [Performance Optimization](./docs/performance)

## Examples

Check out the [examples](./examples) directory for more:

- [MNIST Classification](./examples/mnist.rs)
- [ResNet Implementation](./examples/resnet.rs)
- [GPT-2 from Scratch](./examples/gpt2.rs)
- [Distributed Training](./examples/distributed.rs)
- [WASM Deployment](./examples/wasm)

## Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

## Roadmap

See [TODO.md](./TODO.md) for our detailed development roadmap.

### Current Status: v0.1.0-alpha

- [x] Core tensor operations with 123+ passing tests
- [x] Complete autograd system with gradient computation
- [x] CPU backend with SIMD optimizations
- [x] Complete neural network modules (Linear, Conv2D, pooling, normalization, activations)
- [x] Optimizers (SGD, Adam, AdamW, AdaGrad, RMSprop) with full algorithm implementations
- [x] Data loading framework with parallel processing
- [x] Model serialization with SafeTensors support
- [ ] CUDA backend (in progress)
- [ ] Python bindings

## License

ToRSh is dual-licensed under:

- MIT License ([LICENSE-MIT](./LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))

You may choose either license for your use.

## Acknowledgments

ToRSh builds upon the excellent work of:

- [scirs2](https://github.com/scirs/scirs2) - Scientific computing primitives
- [numrs2](https://github.com/numrs/numrs2) - Numerical computing library
- [Burn](https://github.com/burn-rs/burn) - Inspiration for Rust ML frameworks
- [Candle](https://github.com/huggingface/candle) - Lightweight tensor operations

## Community

- [Discord](https://discord.gg/torsh)
- [GitHub Discussions](https://github.com/cool-japan/torsh/discussions)
- [Twitter](https://twitter.com/torsh_ml)

---

<div align="center">
Built with ❤️ in Rust
</div>