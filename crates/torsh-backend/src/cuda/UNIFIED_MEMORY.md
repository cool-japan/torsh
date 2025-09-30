# CUDA Unified Memory Support

This document describes the unified memory implementation for the ToRSh CUDA backend, which provides seamless data sharing between CPU and GPU without explicit memory transfers.

## Overview

Unified Memory (also known as Managed Memory) is a CUDA feature that creates a pool of managed memory that is accessible from both the CPU and GPU. The memory manager automatically migrates data between host and device as needed, simplifying memory management and reducing explicit copy operations.

## Features

### 1. Unified Memory Allocation
- **Automatic Management**: Memory is automatically migrated between CPU and GPU
- **Single Address Space**: Same pointer works on both CPU and GPU
- **Transparent Access**: No need for explicit cudaMemcpy calls

### 2. Memory Prefetching
- **Device Prefetching**: `prefetch_to_device()` - Move data closer to GPU before kernel execution
- **Host Prefetching**: `prefetch_to_host()` - Move data to CPU for host processing
- **Async Operations**: Prefetching happens asynchronously for better performance

### 3. Memory Advice System
Provides performance hints to the CUDA runtime for optimal data placement:

- **SetReadMostly**: Indicates data will be read frequently but written rarely
- **SetPreferredLocation**: Suggests optimal storage location for data
- **SetAccessedBy**: Indicates which devices will access the memory
- **UnsetReadMostly/UnsetPreferredLocation/UnsetAccessedBy**: Remove previously set hints

### 4. Type-Safe Buffer Abstraction
- **UnifiedBuffer<T>**: Type-safe wrapper around unified memory allocations
- **Buffer Trait Compliance**: Implements standard buffer operations
- **Automatic Cleanup**: RAII-style memory management

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ToRSh CUDA Backend                       │
├─────────────────────────────────────────────────────────────┤
│  CudaBackend                                                │
│  ├── allocate_unified()                                     │
│  ├── prefetch_to_device()                                   │
│  ├── prefetch_to_host()                                     │
│  └── set_memory_advice()                                    │
├─────────────────────────────────────────────────────────────┤
│  CudaMemoryManager                                          │
│  ├── UnifiedAllocation                                      │
│  ├── MemoryAdvice                                           │
│  └── Device Capability Detection                            │
├─────────────────────────────────────────────────────────────┤
│  UnifiedBuffer<T>                                           │
│  ├── Type-safe operations                                   │
│  ├── Buffer trait implementation                            │
│  └── Performance optimization methods                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CUDA Runtime                             │
├─────────────────────────────────────────────────────────────┤
│  cudaMallocManaged()                                        │
│  cudaMemPrefetchAsync()                                     │
│  cudaMemAdvise()                                            │
│  cudaFree()                                                 │
└─────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Basic Allocation and Data Transfer

```rust
use torsh_backend::cuda::{CudaBackend, CudaBackendConfig, UnifiedBuffer};
use torsh_core::DType;

// Initialize backend
let config = CudaBackendConfig::default();
let backend = CudaBackend::initialize(config).await?;

// Check unified memory support
if !backend.supports_unified_memory()? {
    return Err("Device does not support unified memory");
}

// Allocate unified buffer
let device = Arc::new(CudaDevice::new(0)?);
let mut buffer = UnifiedBuffer::<f32>::new(device, 1024, DType::F32)?;

// Copy data from host
let data = vec![1.0, 2.0, 3.0, 4.0];
buffer.copy_from_host(&data)?;

// Data is now accessible from both CPU and GPU
```

### Performance Optimization with Prefetching

```rust
// Prefetch data before GPU computation
buffer.prefetch_to_device(Some(0))?;

// Run GPU kernel here...
// kernel<<<blocks, threads>>>(buffer.as_ptr());

// Prefetch back to host for CPU processing
buffer.prefetch_to_host()?;

// Process on CPU...
```

### Memory Advice for Optimization

```rust
// Set as read-mostly for better caching
buffer.set_memory_advice(MemoryAdvice::SetReadMostly, None)?;

// Set preferred location
buffer.set_memory_advice(MemoryAdvice::SetPreferredLocation, Some(0))?;

// Indicate access pattern
buffer.set_memory_advice(MemoryAdvice::SetAccessedBy, Some(0))?;
```

### Advanced Usage with Multiple Devices

```rust
// For multi-GPU scenarios
buffer.set_accessed_by(0)?;  // GPU 0 will access
buffer.set_accessed_by(1)?;  // GPU 1 will also access
buffer.set_preferred_location(0)?;  // Prefer storing on GPU 0
```

## Performance Considerations

### Benefits
1. **Reduced Memory Copies**: Eliminates explicit cudaMemcpy calls
2. **Automatic Optimization**: Runtime handles data placement
3. **Simplified Code**: Single pointer works everywhere
4. **Demand Paging**: Only transfers data when accessed

### Best Practices
1. **Prefetch Before Use**: Always prefetch data before intensive operations
2. **Use Memory Advice**: Provide hints for access patterns
3. **Batch Operations**: Group memory operations together
4. **Monitor Usage**: Profile memory migration patterns

### Potential Overhead
1. **Page Faults**: First access triggers migration
2. **Bandwidth Usage**: Automatic migrations use PCIe bandwidth
3. **Latency**: Initial access may have higher latency

## Device Requirements

### Minimum Requirements
- **Compute Capability**: 3.0 or higher for basic unified memory
- **Compute Capability**: 6.0 or higher for optimal performance
- **Architecture**: Kepler, Maxwell, Pascal, Volta, Turing, Ampere, or newer

### Feature Detection
```rust
let device = CudaDevice::new(0)?;
let supports_unified = device.supports_feature(CudaFeature::ManagedMemory)?;
let supports_addressing = device.supports_feature(CudaFeature::UnifiedAddressing)?;
```

## Error Handling

The implementation provides comprehensive error handling:

```rust
// Check for device support
match backend.supports_unified_memory() {
    Ok(true) => { /* proceed */ },
    Ok(false) => return Err("Device does not support unified memory"),
    Err(e) => return Err(format!("Failed to check support: {}", e)),
}

// Handle allocation failures
let buffer = match backend.allocate_unified(size) {
    Ok(buf) => buf,
    Err(CudaError::OutOfMemory) => {
        // Handle out-of-memory specifically
        return retry_with_smaller_allocation();
    },
    Err(e) => return Err(format!("Allocation failed: {}", e)),
};
```

## Integration with Existing Code

### Migration from Explicit Memory Management
```rust
// Old approach with explicit copies
let mut device_data = backend.create_buffer::<f32>(size, DType::F32)?;
device_data.copy_from_host(&host_data)?;
// ... GPU work ...
device_data.copy_to_host(&mut result_data)?;

// New approach with unified memory
let mut unified_data = backend.allocate_unified(size)?;
unified_data.copy_from_host(&host_data)?;
unified_data.prefetch_to_device(None)?;
// ... GPU work (direct access to unified_data) ...
unified_data.prefetch_to_host()?;
// ... CPU work (direct access to unified_data) ...
```

### Compatibility with Existing Buffers
UnifiedBuffer implements the same Buffer trait as CudaBuffer, ensuring compatibility:

```rust
fn process_buffer<T>(buffer: &mut dyn Buffer<T>) -> Result<(), BackendError> {
    // Works with both CudaBuffer and UnifiedBuffer
    buffer.fill(T::default())?;
    Ok(())
}
```

## Testing and Validation

The implementation includes comprehensive tests:

1. **Capability Detection**: Verify device support
2. **Allocation/Deallocation**: Memory lifecycle management
3. **Data Integrity**: Ensure data consistency across transfers
4. **Performance Hints**: Validate memory advice operations
5. **Error Conditions**: Test failure scenarios

## Future Enhancements

### Planned Features
1. **Memory Pool Integration**: Integrate with existing memory pool system
2. **Performance Profiling**: Built-in migration tracking
3. **Multi-GPU Optimization**: Cross-GPU unified memory support
4. **Stream Integration**: Async operations with CUDA streams

### Research Areas
1. **Automatic Prefetching**: Machine learning-based prefetch prediction
2. **Access Pattern Analysis**: Runtime optimization based on usage patterns
3. **Memory Compression**: Compressed unified memory for larger datasets

## Conclusion

The unified memory implementation provides a powerful abstraction for seamless CPU-GPU data sharing while maintaining performance through prefetching and memory advice. This feature significantly simplifies memory management in ToRSh applications while providing fine-grained control for performance optimization.