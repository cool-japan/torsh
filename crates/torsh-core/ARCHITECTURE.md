# torsh-core Architecture

This document describes the architecture of the `torsh-core` crate, the foundational layer of the ToRSh deep learning framework.

## Table of Contents

- [Overview](#overview)
- [Core Principles](#core-principles)
- [Module Organization](#module-organization)
- [Component Relationships](#component-relationships)
- [Key Design Patterns](#key-design-patterns)
- [Extension Points](#extension-points)
- [Performance Considerations](#performance-considerations)

## Overview

`torsh-core` provides the fundamental building blocks for the ToRSh framework:

- **Type System**: DType, Shape, and type promotion
- **Device Abstraction**: Platform-independent device representation
- **Error Handling**: Comprehensive error system with context
- **Memory Management**: Efficient memory allocation and pooling
- **Storage Backends**: Unified interface for different memory layouts
- **Debugging Tools**: Runtime introspection and profiling

### Design Philosophy

1. **Zero-cost abstractions**: Performance critical paths have minimal overhead
2. **Type safety**: Compile-time and runtime validation
3. **Extensibility**: Easy to add new devices, dtypes, and backends
4. **SciRS2 Integration**: Deep integration with the scirs2 ecosystem
5. **Production-ready**: Comprehensive error handling and debugging tools

## Core Principles

### 1. Modular Design

Each major component is isolated in its own module with clear interfaces:

```
torsh-core/
├── dtype/          # Data type system
├── shape/          # Tensor shape management
├── device/         # Device abstraction
├── error/          # Error handling
├── storage/        # Memory management
└── ...
```

### 2. Layered Architecture

```
┌─────────────────────────────────────────┐
│     High-Level APIs & Utilities         │  Examples, profiling, debugging
├─────────────────────────────────────────┤
│        Core Abstractions                │  DType, Shape, Device
├─────────────────────────────────────────┤
│      Memory & Storage Layer             │  Allocators, pooling, NUMA
├─────────────────────────────────────────┤
│     Platform-Specific Backends          │  CPU, CUDA, Metal, WebGPU
└─────────────────────────────────────────┘
```

### 3. Separation of Concerns

- **Types** (DType, Shape) are pure data structures
- **Devices** provide computational capabilities
- **Storage** manages memory allocation
- **Errors** handle all failure modes
- **Utilities** add debugging and profiling

## Module Organization

### Core Types Module Graph

```
dtype.rs ──────┐
              ├──> TensorElement ──> Operations
shape.rs ─────┤
              └──> Validation ──────> Error Handling
device.rs ─────────────────────────> Backend Selection
```

### Data Type System (`dtype/`)

```rust
pub enum DType {
    // Integer types
    U8, I8, I16, I32, I64,
    // Float types
    F16, BF16, F32, F64,
    // Complex types
    C64, C128,
    // Quantized types
    QInt8, QUInt8,
}
```

**Key Features:**
- Type promotion system for mixed-precision operations
- IEEE 754 compliance checking
- Custom data type support through traits
- Automatic type conversion with safety checks

**Dependencies:**
- Uses `scirs2_core::numeric` for numerical traits
- Integrates with `scirs2_core::ndarray` for array operations

### Shape Management (`shape/`)

```
┌────────────────┐
│  Shape (Core)  │
└────────┬───────┘
         │
    ┌────┴────┬──────────┬─────────────┐
    │         │          │             │
┌───▼───┐ ┌──▼──┐  ┌────▼─────┐  ┌───▼────┐
│Stride │ │Cache│  │Validation│  │ Utils  │
│       │ │     │  │          │  │        │
└───────┘ └─────┘  └──────────┘  └────────┘
```

**Components:**
- `shape.rs`: Core shape representation with dimension tracking
- `shape_utils.rs`: Common shape operations and patterns
- `shape_validation.rs`: Validation with visual error messages
- `shape_debug.rs`: ASCII visualization and debugging

**Design Decisions:**
- Shapes are immutable for thread safety
- Stride caching for performance (thread-local + global)
- Symbolic shape support for dynamic graphs

### Device Abstraction (`device/`)

```
                    ┌─────────────┐
                    │   Device    │
                    │   (Trait)   │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌─────▼────┐     ┌────▼─────┐
    │  CPU    │      │   CUDA   │     │  Metal   │
    │         │      │          │     │          │
    └─────────┘      └──────────┘     └──────────┘
```

**Submodules:**
- `device/core.rs`: Device trait and base implementations
- `device/capabilities.rs`: Feature detection and scoring
- `device/discovery.rs`: Automatic device selection
- `device/management.rs`: Device pools and health monitoring
- `device/phantom.rs`: Type-level device tracking

**Phantom Types for Compile-Time Safety:**

```rust
// Compile-time device type checking
let tensor: Tensor<CpuDevice, F32> = ...;
let gpu_tensor: Tensor<CudaDevice, F32> = ...;

// This won't compile:
// let result = tensor + gpu_tensor; // Error: device mismatch!

// Type-safe device groups
let devices: DeviceGroup<CudaDevice, 4> = ...;
```

### Error Handling (`error/`)

```
                  ┌──────────────┐
                  │  TorshError  │
                  └───────┬──────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼─────┐     ┌─────▼────┐    ┌──────▼──────┐
   │  Shape   │     │  Index   │    │   General   │
   │  Error   │     │  Error   │    │    Error    │
   └──────────┘     └──────────┘    └─────────────┘
```

**Features:**
- Modular error types (shape, index, general)
- Rich error context with stack traces
- Standard error codes for FFI interoperability
- Error recovery mechanisms
- Source location tracking

**Error Code Mapping:**

```rust
// ToRSh errors map to standard POSIX-like codes
TorshError::OutOfMemory     -> ENOMEM (12)
TorshError::InvalidArgument -> EINVAL (22)
TorshError::NotImplemented  -> ENOSYS (38)

// Custom codes for framework-specific errors
TorshError::ShapeMismatch   -> 1001
TorshError::DTypeMismatch   -> 1011
TorshError::DeviceError     -> 1021
```

### Storage System (`storage/`)

```
┌──────────────────────────────────┐
│   Storage Trait (Abstract)       │
└────────────┬─────────────────────┘
             │
    ┌────────┴────────┬────────────┬──────────┐
    │                 │            │          │
┌───▼────┐   ┌───────▼──┐   ┌─────▼───┐  ┌──▼─────┐
│Aligned │   │  NUMA    │   │ Mapped  │  │  Pool  │
│        │   │          │   │ Storage │  │        │
└────────┘   └──────────┘   └─────────┘  └────────┘
```

**Memory Management Strategies:**

1. **Aligned Storage**: SIMD-friendly memory alignment
2. **NUMA-Aware**: Optimize for multi-socket systems
3. **Memory-Mapped**: Lazy loading for large tensors
4. **Memory Pooling**: Reduce allocation overhead

**Registry Pattern:**

```rust
// Register custom allocators
registry.register(
    "custom_allocator",
    AllocatorMetadata { ... },
    Box::new(MyAllocator::new())
);

// Automatic allocator selection
let allocator = registry.find_best_for_backend(backend_type);
```

## Component Relationships

### Data Flow: Tensor Operation

```
┌──────────┐
│   User   │
└────┬─────┘
     │ operation()
     ▼
┌────────────────┐
│  Validation    │  ◄── Shape, DType checks
└────┬───────────┘
     │ validated
     ▼
┌────────────────┐
│ Device Select  │  ◄── Device capabilities
└────┬───────────┘
     │ device chosen
     ▼
┌────────────────┐
│ Memory Alloc   │  ◄── Storage backend
└────┬───────────┘
     │ memory ready
     ▼
┌────────────────┐
│  Computation   │  ◄── Backend execution
└────┬───────────┘
     │ result
     ▼
┌────────────────┐
│    Return      │
└────────────────┘
```

### Type Promotion Flow

```
Operation(tensor_f32, tensor_i32)
         │
         ▼
┌─────────────────────┐
│  Type Compatibility │
│       Check         │
└──────────┬──────────┘
           │
           ▼ (incompatible)
┌─────────────────────┐
│   Type Promotion    │
│   f32 + i32 → f32   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Execute Operation  │
└─────────────────────┘
```

### Device Discovery & Selection

```
┌─────────────────┐
│ Discover Devices│
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Query Capabilities │  ◄── SIMD, memory, etc.
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Score Performance  │  ◄── Workload profile
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Select Best Device │
└─────────────────────┘
```

## Key Design Patterns

### 1. Builder Pattern

Used extensively for configuration:

```rust
let config = RuntimeConfig::builder()
    .debug_level(DebugLevel::Verbose)
    .validation_level(ValidationLevel::Strict)
    .enable_profiling(true)
    .build();
```

### 2. Registry Pattern

For extensible component registration:

```rust
// Device registry
DeviceRegistry::register(device_type, factory);

// Allocator registry
AllocatorRegistry::register(name, metadata, allocator);
```

### 3. Phantom Types

For compile-time type safety:

```rust
struct Tensor<D: PhantomDevice, T: DType> {
    data: Storage,
    _phantom: PhantomData<(D, T)>,
}
```

### 4. Strategy Pattern

For algorithm selection:

```rust
trait AllocationStrategy {
    fn allocate(&self, size: usize) -> Result<*mut u8>;
}

// Different strategies: NUMA, pooled, aligned
```

### 5. Observer Pattern

For monitoring and telemetry:

```rust
// Performance profiler observes operations
profiler.record_operation("matmul", duration);

// Memory debugger tracks allocations
debugger.record_allocation(size, layout);
```

### 6. Flyweight Pattern

For shape stride caching:

```rust
// Reuse computed strides across tensors
let strides = STRIDE_CACHE.get_or_compute(shape);
```

## Extension Points

### Adding a New Data Type

1. Define the type in `dtype/extended.rs`
2. Implement `TensorElement` trait
3. Add to `DType` enum
4. Implement type promotion rules
5. Add test cases

### Adding a New Device Backend

1. Implement `Device` trait in `device/implementations.rs`
2. Add device capabilities
3. Register device factory
4. Implement memory allocator
5. Add backend-specific optimizations

### Adding Custom Storage

1. Implement `Storage` trait
2. Register allocator in registry
3. Specify allocation requirements
4. Add metadata for discovery

## Performance Considerations

### Hot Paths

1. **Tensor indexing**: Uses raw pointers, bounds checking only in debug
2. **Shape validation**: Cached strides, thread-local caches
3. **Type promotion**: Compile-time when possible, minimal runtime overhead
4. **Memory allocation**: Pooled for small tensors, aligned for SIMD

### SIMD Optimization

```rust
#[cfg(target_feature = "avx2")]
fn simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::*;
    // AVX2 vectorized implementation
}

#[cfg(target_feature = "neon")]
fn simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    use std::arch::aarch64::*;
    // NEON vectorized implementation
}
```

### Memory Layout Optimization

- **C-contiguous**: Default, best for row-major operations
- **F-contiguous**: Better for column-major operations
- **Strided**: Flexible but slower
- **Aligned**: 32/64-byte alignment for SIMD

### Cache Efficiency

```rust
// Thread-local stride cache
thread_local! {
    static STRIDE_CACHE: RefCell<HashMap<Shape, Vec<usize>>> = ...;
}

// Global LRU cache with eviction
static GLOBAL_STRIDE_CACHE: Lazy<Mutex<LruCache<...>>> = ...;
```

## Runtime Configuration

### Debug Levels

```rust
pub enum DebugLevel {
    None,       // No debug output
    Essential,  // Critical errors only
    Standard,   // Normal debug info
    Verbose,    // Detailed debug info
    Paranoid,   // Everything, including internals
}
```

### Validation Levels

```rust
pub enum ValidationLevel {
    Essential,  // Only check critical invariants
    Standard,   // Normal validation
    Strict,     // Thorough validation
    Maximum,    // Every possible check
}
```

### Configuration Presets

- **Development**: Verbose debugging, strict validation
- **Testing**: Standard debugging, strict validation
- **Production**: Essential debugging, essential validation
- **Profiling**: Minimal debugging, standard validation

## Testing Strategy

### Unit Tests

- Per-module tests in `#[cfg(test)]` blocks
- Cover edge cases and error conditions
- Property-based testing with `proptest`

### Integration Tests

- Backend integration tests
- Cross-module interaction tests
- SciRS2 integration verification

### Benchmark Tests

- Criterion benchmarks in `benches/`
- Performance regression detection
- Platform-specific optimizations

### Fuzz Testing

- Cargo-fuzz targets for shape operations
- Random input generation
- Invariant checking

## Future Directions

### Planned Enhancements

1. **Graph-based shape inference** for optimization
2. **Automatic memory layout optimization**
3. **Distributed tensor metadata management**
4. **Enhanced compile-time type checking**
5. **WebGPU compute shader integration**

### Research Topics

1. Cache-oblivious algorithms for shape operations
2. Tensor expression templates for optimization
3. Type-level automatic differentiation
4. Neuromorphic computing data structures

## References

- [PyTorch Tensor Implementation](https://pytorch.org/)
- [TensorFlow Core](https://www.tensorflow.org/)
- [ndarray Rust Crate](https://docs.rs/ndarray/)
- [SciRS2 Documentation](https://github.com/cool-japan/scirs)
- [IEEE 754 Floating-Point Standard](https://en.wikipedia.org/wiki/IEEE_754)

## Contributing

When contributing to torsh-core, please:

1. Follow the module organization patterns
2. Add comprehensive tests for new features
3. Update this architecture document
4. Maintain zero-cost abstractions
5. Ensure SciRS2 POLICY compliance

---

*Last Updated: 2025-10-23*
*Version: 0.1.0-alpha.2*
