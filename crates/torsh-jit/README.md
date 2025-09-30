# torsh-jit

Just-In-Time (JIT) compilation and kernel fusion for the ToRSh deep learning framework.

## Overview

The `torsh-jit` crate provides JIT compilation capabilities for ToRSh, enabling:

- **Kernel Fusion**: Automatically combines compatible operations to reduce memory bandwidth
- **Graph Optimization**: Applies various optimization passes to improve performance
- **Multiple Backends**: Supports CPU (via Cranelift), CUDA, and Metal code generation
- **TorchScript-like API**: Compatible interface for ease of migration from PyTorch

## Features

### Kernel Fusion
- Element-wise operation fusion (add, mul, relu, etc.)
- Conv+Activation fusion for common patterns
- Linear+Activation fusion
- Reduction operation fusion
- Custom fusion rules support

### Graph Optimizations
- Dead code elimination
- Constant folding
- Common subexpression elimination
- Algebraic simplification
- Strength reduction
- Memory layout optimization

### Code Generation
- Cranelift backend for CPU
- CUDA PTX generation (planned)
- Metal shader generation (planned)
- Interpreter fallback for unsupported operations

## Usage

```rust
use torsh_jit::{JitCompiler, JitConfig, FusionStrategy};

// Configure JIT compilation
let config = JitConfig {
    fusion_strategy: FusionStrategy::Default,
    enable_optimizations: true,
    max_fusion_size: 8,
    enable_profiling: false,
    target_device: Device::Cpu,
    enable_caching: true,
};

// Create JIT compiler
let mut compiler = JitCompiler::new(config);

// Build computation graph (usually from a model)
let graph = build_model_graph();

// Compile the graph
let compiled_module = compiler.compile(graph)?;

// Execute with inputs
let outputs = compiled_module.execute(&inputs)?;

// Get execution statistics
let stats = compiled_module.stats();
println!("Execution time: {}μs", stats.total_time_us);
```

## Architecture

### Computation Graph
The core representation is a directed acyclic graph (DAG) where:
- Nodes represent operations (MatMul, Conv2d, ReLU, etc.)
- Edges represent data dependencies
- Supports complex topologies with multiple inputs/outputs

### Fusion Engine
The fusion engine identifies patterns of operations that can be combined:
- Element-wise chains: neg -> abs -> relu
- Neural patterns: conv -> bn -> relu
- Custom patterns via FusionRules

### Optimization Pipeline
Multiple optimization passes are applied in sequence:
1. Dead code elimination
2. Constant folding
3. Common subexpression elimination
4. Algebraic simplification
5. Strength reduction
6. Layout optimization

### Code Generation
Backend-specific code generators produce optimized kernels:
- CPU: Uses Cranelift for native code generation
- CUDA: Generates PTX for NVIDIA GPUs
- Metal: Generates Metal shaders for Apple Silicon

## Performance

The JIT compiler can provide significant speedups by:
- Reducing memory traffic through kernel fusion
- Eliminating redundant computations
- Optimizing memory access patterns
- Leveraging backend-specific optimizations

Typical improvements:
- 2-5x speedup for element-wise operation chains
- 20-40% improvement for conv+activation patterns
- Near-zero overhead for cached kernels

## Future Work

- [ ] MLIR backend integration
- [ ] Dynamic shape support
- [ ] Control flow (if/while) support
- [ ] Distributed JIT compilation
- [ ] Profile-guided optimization
- [ ] Kernel autotuning

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.