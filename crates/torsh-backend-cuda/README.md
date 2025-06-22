# torsh-backend-cuda

CUDA backend for the ToRSh deep learning framework.

## Features

- High-performance GPU tensor operations
- Custom CUDA kernels for optimal performance
- Integration with cuBLAS and cuDNN
- Memory pooling for efficient allocation
- Stream-based asynchronous execution
- Integration with scirs2 GPU ecosystem

## Requirements

- NVIDIA GPU with compute capability 5.0 or higher
- CUDA Toolkit 11.0 or higher
- cuDNN 8.0 or higher (optional, for optimized neural network operations)

## Usage

```rust
use torsh_backend_cuda::{CudaBackend, CudaBackendConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create CUDA backend
    let config = CudaBackendConfig {
        device_id: 0,
        allow_tf32: true,
        ..Default::default()
    };
    
    let backend = CudaBackend::initialize(config).await?;
    
    // Create tensors and perform operations
    let a = backend.create_buffer::<f32>(1024, torsh_core::DType::F32)?;
    let b = backend.create_buffer::<f32>(1024, torsh_core::DType::F32)?;
    let mut result = backend.create_buffer::<f32>(1024, torsh_core::DType::F32)?;
    
    // Perform addition on GPU
    backend.add_tensors(&*a, &*b, &mut *result)?;
    
    Ok(())
}
```

## Environment Variables

- `CUDA_PATH` or `CUDA_HOME`: Path to CUDA installation
- `CUDNN_PATH`: Path to cuDNN installation (if different from CUDA)

## Performance

The CUDA backend provides significant performance improvements over CPU operations:

- Elementwise operations: 10-50x speedup
- Matrix multiplication: 20-100x speedup  
- Convolution operations: 30-200x speedup
- Memory bandwidth: Up to 900 GB/s on modern GPUs

## Memory Management

The backend includes sophisticated memory management:

- **Memory pooling**: Reduces allocation overhead
- **Stream-based execution**: Enables overlapping computation and memory transfers
- **Unified memory**: Automatic data migration between CPU and GPU
- **Memory monitoring**: Track usage and detect leaks

## Kernel Implementation

Custom CUDA kernels are implemented for optimal performance:

- **Elementwise operations**: Vectorized operations using SIMD instructions
- **Reduction operations**: Hierarchical reduction with shared memory
- **Matrix operations**: Optimized memory access patterns
- **Neural network operations**: Fused kernels for common patterns

## Testing

Run tests with CUDA device:

```bash
cargo test --features cuda
```

Note: Tests require a CUDA-capable GPU to run successfully.

## Contributing

See the main ToRSh contributing guidelines. When adding CUDA kernels:

1. Add kernel implementation in `src/kernels/*.cu`
2. Add Rust bindings in `src/kernels/*.rs`  
3. Add tests with actual GPU verification
4. Update documentation with performance characteristics

## License

Licensed under MIT OR Apache-2.0, same as the main ToRSh project.