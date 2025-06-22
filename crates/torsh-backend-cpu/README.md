# ToRSh CPU Backend

High-performance CPU backend implementation for the ToRSh deep learning framework.

## Features

- **Multi-threading**: Parallel tensor operations using Rayon thread pool
- **SIMD Operations**: Vectorized computations when `simd` feature is enabled
- **Memory Management**: Efficient memory pooling and allocation strategies
- **Kernel Execution**: Built-in implementations for common tensor operations
- **Profiling**: Comprehensive performance monitoring and Chrome trace export
- **Cross-platform**: Works on Windows, macOS, and Linux

## Architecture

The CPU backend provides:

- **CpuBackend**: Main backend implementation managing devices and execution
- **CpuDevice**: Represents CPU compute resources with core count and capabilities
- **CpuBuffer**: Thread-safe memory management using system RAM
- **CpuKernel**: Compiled kernel functions for tensor operations
- **CpuMemoryManager**: Memory pooling with size-class allocation
- **CpuProfiler**: Performance analysis with event tracking

## Performance Features

### Multi-threading with Rayon

When the `rayon-threads` feature is enabled (default), tensor operations are automatically parallelized across available CPU cores:

```rust
// Element-wise operations are automatically parallelized
let result = tensor_a.add(&tensor_b)?;
```

### SIMD Vectorization

Enable the `simd` feature for vectorized operations on supported data types:

```toml
[dependencies]
torsh-backend-cpu = { version = "0.1", features = ["simd"] }
```

SIMD operations provide significant speedups for:
- Element-wise arithmetic (add, mul, sub, div)
- Activation functions (ReLU, sigmoid, tanh)
- Reduction operations (sum, dot product)

### BLAS Integration

Optional BLAS backend for optimized linear algebra:

```toml
[dependencies]
torsh-backend-cpu = { version = "0.1", features = ["blas"] }
```

On macOS, this uses the Accelerate framework for maximum performance.

## Supported Operations

The CPU backend includes optimized implementations for:

### Element-wise Operations
- Addition, subtraction, multiplication, division
- Power operations
- Comparison operations

### Activation Functions
- ReLU, Sigmoid, Tanh
- Softmax, LogSoftmax

### Linear Algebra
- Matrix multiplication (2D and batched)
- Transpose operations
- Vector operations

### Reduction Operations
- Sum, mean, max, min
- Argmax, argmin

## Usage

```rust
use torsh_backend_cpu::prelude::*;

// Initialize the CPU backend
torsh_backend_cpu::init()?;

// Create a CPU device
let backend = CpuBackend::new()?;
let device = backend.default_device()?;

// Create buffers
let buffer_desc = BufferDescriptor::new(1024, BufferUsage::STORAGE);
let buffer = backend.create_buffer(&device, &buffer_desc)?;

// Execute kernels
let kernel_desc = KernelDescriptor {
    name: "add".to_string(),
    entry_point: "main".to_string(),
    source: None,
    bytecode: None,
};
let kernel = backend.create_kernel(&device, &kernel_desc)?;
```

## Memory Management

The CPU backend uses a sophisticated memory management system:

- **Size-class allocation**: Groups allocations by size for better reuse
- **Memory pooling**: Reuses deallocated blocks to reduce allocation overhead
- **Thread-safe**: All operations are safe for concurrent access
- **Statistics tracking**: Monitors allocation patterns for optimization

## Profiling

Built-in profiling support for performance analysis:

```rust
let profiler = backend.profiler()?;
profiler.start_profiling()?;

// Run operations...

profiler.stop_profiling()?;
let stats = profiler.get_stats()?;
profiler.export_chrome_trace("trace.json")?;
```

## Platform-specific Optimizations

### macOS
- Uses Accelerate framework when `blas` feature is enabled
- Optimized memory bandwidth detection
- ARM64 NEON instructions on Apple Silicon

### Linux
- Detects available CPU features from /proc/cpuinfo
- Uses system memory information from /proc/meminfo
- Supports both x86_64 and ARM architectures

### Windows
- Compatible with MSVC and MinGW toolchains
- Uses Windows API for system information
- Supports AVX/SSE instruction sets

## Configuration

### Feature Flags

- `rayon-threads` (default): Enable multi-threading with Rayon
- `simd`: Enable SIMD vectorization
- `blas`: Enable BLAS acceleration (macOS Accelerate)

### Environment Variables

- `TORSH_CPU_THREADS`: Override automatic thread detection
- `TORSH_CPU_SIMD`: Force enable/disable SIMD (true/false)
- `TORSH_CPU_PROFILING`: Enable profiling by default (true/false)

## Performance Tips

1. **Enable SIMD**: Use the `simd` feature for vectorized operations
2. **BLAS on macOS**: Enable `blas` feature for linear algebra acceleration
3. **Thread count**: Set `TORSH_CPU_THREADS` to match your workload
4. **Memory alignment**: Use 64-byte alignment for cache efficiency
5. **Batch operations**: Process data in batches for better parallelization

## Limitations

- Single precision (f32) is preferred for SIMD operations
- Very small tensors may not benefit from parallelization
- Memory usage scales with thread count for parallel operations
- BLAS integration currently only available on macOS

## Contributing

When adding new kernel implementations:

1. Add the kernel function to `CpuKernel::compile_kernel()`
2. Implement both sequential and parallel versions
3. Add SIMD variants when beneficial
4. Include comprehensive tests
5. Update documentation

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.