# torsh-backends

Unified backend implementation for ToRSh with PyTorch-compatible API, leveraging SciRS2's GPU acceleration.

## Overview

This crate provides a unified backend system that integrates with SciRS2's compute backends:

- **CPU Backend**: Optimized CPU operations with SIMD and parallelism
- **CUDA Backend**: NVIDIA GPU acceleration via scirs2-core's CUDA support
- **Metal Backend**: Apple GPU acceleration via scirs2-core's Metal/MPS support  
- **ROCm Backend**: AMD GPU acceleration (via scirs2-core when available)
- **WebGPU Backend**: Cross-platform GPU support (via scirs2-core when available)

Note: All backend implementations are unified in this single crate using feature flags, eliminating the need for separate torsh-backend-* crates.

## Architecture

The backend system leverages SciRS2's GPU infrastructure:

```rust
use torsh_backends::{Backend, BackendType, Device};

// Unified backend with runtime selection
let backend = Backend::new(BackendType::Auto)?;  // Auto-detect best backend
let backend = Backend::new(BackendType::Cuda)?;  // Explicit CUDA
let backend = Backend::new(BackendType::Metal)?; // Explicit Metal
```

## Feature Flags

```toml
[dependencies]
torsh-backends = { version = "0.1.0-alpha.2", features = ["cuda", "metal"] }

# Available features:
# - "cpu" (default): CPU backend with SIMD optimizations
# - "cuda": NVIDIA GPU backend via scirs2-core
# - "metal": Apple GPU backend via scirs2-core
# - "rocm": AMD GPU backend via scirs2-core
# - "webgpu": WebGPU backend via scirs2-core
```

## Usage

### Unified Backend API

```rust
use torsh_backends::prelude::*;

// Automatic backend selection based on availability
let backend = Backend::auto()?;

// Query available backends
for backend_type in Backend::available() {
    println!("Available: {:?}", backend_type);
}

// Create backend with specific configuration
let backend = BackendBuilder::new()
    .backend_type(BackendType::Cuda)
    .device_id(0)
    .memory_pool_size(4 * 1024 * 1024 * 1024)  // 4GB
    .enable_tensor_cores(true)
    .build()?;

// All backends use the same API
let a = backend.randn(&[1024, 1024], DType::F32)?;
let b = backend.randn(&[1024, 1024], DType::F32)?;
let c = backend.matmul(&a, &b)?;
```

### CPU Backend

```rust
// CPU backend leverages scirs2-core's optimized operations
let cpu_backend = Backend::cpu()
    .num_threads(8)
    .enable_simd(true)
    .build()?;

// Uses OpenBLAS/MKL/Accelerate via scirs2
let result = cpu_backend.gemm(&a, &b, 1.0, &c, 0.0)?;
```

### CUDA Backend  

```rust
#[cfg(feature = "cuda")]
{
    // CUDA backend via scirs2-core's CUDA kernels
    let cuda_backend = Backend::cuda()
        .device(0)
        .enable_cudnn(true)
        .enable_tensor_cores(true)
        .build()?;
    
    // Async execution with streams
    let stream = cuda_backend.create_stream()?;
    cuda_backend.matmul_async(&a, &b, &stream).await?;
}
```

### Metal Backend

```rust  
#[cfg(feature = "metal")]
{
    // Metal backend with MPS via scirs2-core
    let metal_backend = Backend::metal()
        .enable_mps(true)  // Metal Performance Shaders
        .build()?;
    
    // Leverages Apple's optimized kernels
    let result = metal_backend.conv2d(&input, &kernel, ConvConfig {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
        groups: 1,
    })?;
}
```

### Unified Operations

```rust
// All backends support the same operations via scirs2
impl Backend {
    // BLAS operations (via scirs2-core)
    pub fn gemm(&self, a: &Tensor, b: &Tensor, alpha: f32, 
                c: &Tensor, beta: f32) -> Result<Tensor>;
    pub fn gemv(&self, a: &Tensor, x: &Tensor, alpha: f32,
                y: &Tensor, beta: f32) -> Result<Tensor>;
    
    // DNN operations (via scirs2's GPU kernels)
    pub fn conv2d(&self, input: &Tensor, weight: &Tensor,
                  config: ConvConfig) -> Result<Tensor>;
    pub fn batch_norm(&self, input: &Tensor, mean: &Tensor,
                      var: &Tensor, training: bool) -> Result<Tensor>;
    
    // Optimized fused operations from scirs2
    pub fn fused_adam_step(&self, params: &mut [Tensor], 
                           grads: &[Tensor], state: &mut AdamState,
                           lr: f32, betas: (f32, f32)) -> Result<()>;
}
```

## Device Management

```rust
// Unified device abstraction
let devices = Backend::list_devices()?;
for device in devices {
    println!("{}: {} ({}GB memory)", 
             device.id(), device.name(), device.memory_gb());
}

// Multi-device support
let backend_gpu0 = Backend::new(BackendType::Cuda)?.device(0)?;
let backend_gpu1 = Backend::new(BackendType::Cuda)?.device(1)?;

// Device synchronization
backend.synchronize()?;
```

## Memory Management

```rust
// Unified memory pool leveraging scirs2's allocators
let backend = Backend::new(BackendType::Auto)?
    .memory_pool(MemoryPoolConfig {
        initial_size: 1 << 30,      // 1GB
        max_size: 4 << 30,          // 4GB  
        strategy: AllocationStrategy::BestFit,
        enable_defrag: true,
    })?;

// Zero-copy host-device transfers (when supported)
let pinned = backend.alloc_pinned(&[1024, 1024], DType::F32)?;
backend.copy_host_to_device_async(&host_data, &mut pinned).await?;
```

## Performance Features

```rust
// Auto-tuning (via scirs2's auto-tuning infrastructure)
let backend = Backend::new(BackendType::Cuda)?
    .enable_autotuning(true)
    .autotuning_cache_file("tuning_cache.json")?;

// Mixed precision training
let backend = backend.enable_mixed_precision(MixedPrecisionConfig {
    compute_dtype: DType::F16,
    accum_dtype: DType::F32,
    scale_factor: 65536.0,
})?;

// Graph optimization (when using CUDA)
#[cfg(feature = "cuda")]
let graph = backend.capture_graph(|| {
    let x = backend.matmul(&a, &b)?;
    let y = backend.relu(&x)?;
    backend.matmul(&y, &c)
})?;
let result = backend.launch_graph(&graph, &inputs)?;
```

## Integration with SciRS2

This crate fully leverages SciRS2's backend infrastructure:

### Backend Implementation Status

| Backend | SciRS2 Integration | Features |
|---------|-------------------|----------|
| CPU | ✅ scirs2-core (default) | OpenBLAS/MKL/Accelerate, SIMD, Rayon parallelism |
| CUDA | ✅ scirs2-core with `cuda` | Optimized kernels, cuDNN, Tensor Cores, Streams |
| Metal | ✅ scirs2-core with `metal` | Metal Performance Shaders, Unified memory |
| ROCm | 🚧 scirs2-core with `rocm` | Available when scirs2 implements |
| WebGPU | 🚧 scirs2-core with `wgpu` | Available when scirs2 implements |

### Leveraged SciRS2 Features

- **GPU Kernels**: All GPU operations use scirs2's optimized kernels
- **Auto-tuning**: Kernel selection via scirs2's auto-tuning
- **Memory Management**: Efficient pooling from scirs2
- **Async Execution**: Built on scirs2's async GPU model
- **BLAS/LAPACK**: CPU ops via scirs2's math libraries

### Migration from Separate Backend Crates

The previous separate backend crates (`torsh-backend-cpu`, `torsh-backend-cuda`, `torsh-backend-metal`) are now deprecated. Use feature flags instead:

```toml
# Old (deprecated)
torsh-backend-cuda = "0.1.0-alpha.2"

# New (unified)
torsh-backends = { version = "0.1.0-alpha.2", features = ["cuda"] }
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.