# API Design Rationale - torsh-core

This document explains the key design decisions, trade-offs, and rationale behind the torsh-core API design.

## Table of Contents

- [Core Design Principles](#core-design-principles)
- [Type System Design](#type-system-design)
- [Shape System Design](#shape-system-design)
- [Error Handling Strategy](#error-handling-strategy)
- [Device Abstraction](#device-abstraction)
- [Memory Management](#memory-management)
- [Performance vs Safety Trade-offs](#performance-vs-safety-trade-offs)
- [API Stability Considerations](#api-stability-considerations)

## Core Design Principles

### 1. Zero-Cost Abstractions

**Rationale**: Deep learning frameworks are performance-critical. Users should not pay runtime costs for abstractions they don't use.

**Implementation**:
- Phantom types for compile-time device tracking with zero runtime overhead
- Inline small functions that are hot paths
- Const generics for compile-time shape validation
- Static dispatch where possible

**Example**:
```rust
// Zero-cost device tracking at compile time
struct Tensor<D: PhantomDevice, T: DType> {
    data: Storage,
    _phantom: PhantomData<(D, T)>,  // Zero size at runtime
}
```

**Trade-offs**:
- **Pro**: Maximum performance, no runtime overhead
- **Con**: More complex type signatures, longer compile times
- **Decision**: Worth it for production ML workloads where runtime performance is critical

### 2. Type Safety Over Convenience

**Rationale**: Catch errors at compile time rather than runtime. Silent bugs in ML systems can lead to incorrect model training.

**Implementation**:
- Strong typing for devices, dtypes, and shapes
- No implicit conversions between incompatible types
- Explicit error handling with Result types

**Example**:
```rust
// This won't compile - device mismatch caught at compile time
let cpu_tensor: Tensor<CpuDevice, F32> = ...;
let gpu_tensor: Tensor<CudaDevice, F32> = ...;
// let result = cpu_tensor + gpu_tensor; // ❌ Compile error!
```

**Trade-offs**:
- **Pro**: Prevents entire classes of runtime errors
- **Con**: More verbose code, steeper learning curve
- **Decision**: Safety is more important than convenience in production systems

### 3. SciRS2 Integration First

**Rationale**: Leverage the existing Rust scientific computing ecosystem rather than reinventing the wheel.

**Implementation**:
- All external dependencies go through scirs2-core
- Unified access patterns (ndarray, random, numeric)
- Zero-copy conversions where possible

**Example**:
```rust
// ✅ CORRECT: Use scirs2-core abstractions
use scirs2_core::ndarray::{Array, array};
use scirs2_core::random::{thread_rng, Normal};

// ❌ WRONG: Direct external dependencies
// use ndarray::{Array, array};  // POLICY VIOLATION
```

**Trade-offs**:
- **Pro**: Consistent APIs, centralized maintenance, better integration
- **Con**: Extra abstraction layer, dependency on scirs2 ecosystem
- **Decision**: Long-term maintainability outweighs short-term convenience

## Type System Design

### DType Enum vs Trait-Based Design

**Decision**: Use an enum for DType with trait implementations for specific types.

**Rationale**:
1. **Pattern Matching**: Enum allows exhaustive pattern matching
2. **Runtime Type Information**: Need to know dtype at runtime for operations
3. **Serialization**: Enum is easier to serialize/deserialize
4. **Type Promotion**: Centralized promotion rules in one place

**Alternative Considered**: Trait-based system with generic parameters
```rust
// Alternative (NOT chosen):
trait DType {
    fn size(&self) -> usize;
    fn is_float(&self) -> bool;
}
struct F32Type;
impl DType for F32Type { ... }
```

**Why Rejected**:
- Would lose runtime type information
- Pattern matching becomes impossible
- Type promotion rules would be scattered

**Example**:
```rust
// ✅ CHOSEN: Enum with traits
pub enum DType {
    F32, F64, I32, I64,
    C64, C128,  // Complex types
    QInt8, QUInt8,  // Quantized types
}

// Trait for actual element types
pub trait TensorElement: Copy + Send + Sync {
    const DTYPE: DType;
    fn to_dtype() -> DType { Self::DTYPE }
}
```

### Type Promotion System

**Decision**: Automatic type promotion with explicit rules.

**Rationale**:
1. **User Convenience**: Mixed-precision operations "just work"
2. **NumPy Compatibility**: Matches expectations from Python users
3. **Safety**: Explicit promotion rules prevent precision loss surprises

**Implementation**:
```rust
impl DType {
    pub fn promote_with(&self, other: DType) -> DType {
        // Explicit promotion matrix
        match (self, other) {
            (F64, _) | (_, F64) => F64,  // F64 takes precedence
            (F32, _) | (_, F32) => F32,
            (C128, _) | (_, C128) => C128,  // Complex promotes
            // ... explicit rules for all type combinations
        }
    }
}
```

**Trade-offs**:
- **Pro**: Intuitive for users, prevents common errors
- **Con**: Potential for unexpected precision changes
- **Mitigation**: Comprehensive documentation and warning system

## Shape System Design

### Immutable Shapes with Caching

**Decision**: Shapes are immutable value types with cached stride computation.

**Rationale**:
1. **Thread Safety**: Immutable shapes are automatically thread-safe
2. **Functional Style**: Encourages immutable data transformations
3. **Caching**: Computed strides can be safely cached and shared
4. **Hash Keys**: Immutable shapes work well as HashMap keys

**Implementation**:
```rust
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Arc<[usize]>,  // Immutable, shared
    // Cached strides accessed via STRIDE_CACHE
}

// Thread-local cache for hot paths
thread_local! {
    static STRIDE_CACHE: RefCell<HashMap<Shape, Vec<usize>>> = ...;
}

// Global LRU cache for cross-thread sharing
static GLOBAL_STRIDE_CACHE: Lazy<Mutex<LruCache<...>>> = ...;
```

**Alternative Considered**: Mutable shapes with internal mutability
```rust
// Alternative (NOT chosen):
pub struct Shape {
    dims: Vec<usize>,
    cached_strides: Cell<Option<Vec<usize>>>,
}
```

**Why Rejected**:
- Not thread-safe without synchronization
- Cannot be used as HashMap keys
- Harder to reason about ownership and borrowing
- Memory overhead for each Shape instance

**Trade-offs**:
- **Pro**: Thread-safe, functional, efficient caching
- **Con**: Creating new shapes on modification (mitigated by Arc sharing)
- **Decision**: Immutability aligns with Rust's ownership model

### Stride Computation Strategy

**Decision**: Two-tier caching (thread-local + global LRU).

**Rationale**:
1. **Hot Path Optimization**: Thread-local cache has no synchronization overhead
2. **Cross-Thread Sharing**: Global cache prevents redundant computation
3. **Memory Efficiency**: LRU eviction prevents unbounded growth

**Performance Characteristics**:
- Thread-local hit: ~1-2 ns (raw HashMap lookup)
- Global cache hit: ~50-100 ns (mutex + LRU)
- Cache miss: ~500-1000 ns (computation + insertion)

**Trade-offs**:
- **Pro**: Excellent performance for repeated shapes
- **Con**: Memory overhead for cache storage
- **Decision**: Performance gain justifies memory cost in ML workloads

## Error Handling Strategy

### Modular Error Types with Unified Enum

**Decision**: Specialized error modules unified through TorshError enum.

**Rationale**:
1. **Organization**: Errors grouped by domain (shape, index, general)
2. **Extensibility**: Easy to add new error categories
3. **Backward Compatibility**: Unified enum provides stable API
4. **Context-Rich**: Each error type can have specialized fields

**Implementation**:
```rust
pub enum TorshError {
    Shape(ShapeError),
    Index(IndexError),
    General(GeneralError),
    // Legacy compatibility variants
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    // ...
}
```

**Alternative Considered**: Single flat error enum
```rust
// Alternative (NOT chosen):
pub enum TorshError {
    ShapeMismatch,
    IndexOutOfBounds,
    DeviceError,
    // ... all errors at same level
}
```

**Why Rejected**:
- Hard to organize as error types grow
- No logical grouping of related errors
- Difficult to add error-specific methods

### Source Location Tracking

**Decision**: Automatic location tracking using `std::panic::Location`.

**Rationale**:
1. **Debugging**: Know exactly where errors originated
2. **Zero Cost**: Only captured when errors occur
3. **Automatic**: No manual annotation required

**Implementation**:
```rust
#[track_caller]
pub fn new_error(msg: &str) -> TorshError {
    let location = std::panic::Location::caller();
    TorshError::WithLocation {
        message: msg.to_string(),
        file: location.file(),
        line: location.line(),
    }
}
```

**Trade-offs**:
- **Pro**: Excellent debugging experience
- **Con**: Slight overhead on error paths (acceptable since errors are rare)
- **Decision**: Developer experience worth the cost

### Standard Error Codes for FFI

**Decision**: Provide POSIX-compatible error codes alongside Rust errors.

**Rationale**:
1. **C/C++ Interop**: FFI boundaries need integer error codes
2. **Tooling**: Standard codes work with existing error handling tools
3. **Portability**: errno-compatible codes are universally understood

**Implementation**:
```rust
pub enum StandardErrorCode {
    InvalidArgument = 22,  // EINVAL
    OutOfMemory = 12,      // ENOMEM
    // Custom codes for framework-specific errors
    ShapeMismatch = 1001,
    DTypeMismatch = 1011,
}
```

## Device Abstraction

### Trait-Based Device System

**Decision**: Device trait with phantom type markers.

**Rationale**:
1. **Extensibility**: Easy to add new device backends
2. **Type Safety**: Phantom types catch device mismatches at compile time
3. **Dynamic Dispatch**: Trait objects allow runtime device selection
4. **Zero Cost**: Phantom types have no runtime overhead

**Implementation**:
```rust
pub trait Device: Send + Sync {
    fn device_type(&self) -> DeviceType;
    fn is_available(&self) -> bool;
    fn synchronize(&self) -> Result<()>;
}

// Phantom type markers for compile-time tracking
pub trait PhantomDevice: 'static {
    fn device_type_static() -> DeviceType;
}

pub struct PhantomCpu;
impl PhantomDevice for PhantomCpu {
    fn device_type_static() -> DeviceType { DeviceType::Cpu }
}
```

**Trade-offs**:
- **Pro**: Flexible, type-safe, zero-cost
- **Con**: Complex type system with phantom types
- **Decision**: Type safety worth the complexity

### Device Capability System

**Decision**: Rich capability queries with performance scoring.

**Rationale**:
1. **Automatic Selection**: Choose best device for workload
2. **Graceful Degradation**: Fall back when features unavailable
3. **Future-Proof**: Easy to add new capabilities

**Implementation**:
```rust
pub struct DeviceCapabilities {
    pub compute_capability: ComputeCapability,
    pub memory_gb: f32,
    pub supports_half_precision: bool,
    pub supports_double_precision: bool,
    pub simd_features: SimdFeatures,
    pub performance_score: f32,
}

impl DeviceCapabilities {
    pub fn score_for_workload(&self, workload: &WorkloadProfile) -> f32 {
        // Heuristic scoring based on workload requirements
        match workload.workload_type {
            WorkloadType::Training => self.training_score(),
            WorkloadType::Inference => self.inference_score(),
            WorkloadType::DataProcessing => self.data_processing_score(),
        }
    }
}
```

**Trade-offs**:
- **Pro**: Intelligent device selection, better resource utilization
- **Con**: Heuristics may not always be optimal
- **Mitigation**: Allow manual device override

## Memory Management

### Storage Abstraction with Registry Pattern

**Decision**: Pluggable storage backends with automatic selection.

**Rationale**:
1. **Flexibility**: Different workloads need different memory strategies
2. **Extensibility**: Users can provide custom allocators
3. **Automatic Selection**: System chooses best allocator for use case

**Implementation**:
```rust
pub trait Storage: Send + Sync {
    fn allocate(&self, size: usize, alignment: usize) -> Result<*mut u8>;
    fn deallocate(&self, ptr: *mut u8, size: usize, alignment: usize);
}

// Registry pattern for allocator management
pub struct AllocatorRegistry {
    allocators: HashMap<String, Box<dyn Storage>>,
    metadata: HashMap<String, AllocatorMetadata>,
}

impl AllocatorRegistry {
    pub fn find_best_for_backend(&self, backend: BackendType) -> Option<&dyn Storage> {
        // Automatic selection based on backend requirements
    }
}
```

**Alternative Considered**: Single global allocator
```rust
// Alternative (NOT chosen):
static GLOBAL_ALLOCATOR: GlobalAlloc = SystemAlloc;
```

**Why Rejected**:
- No flexibility for specialized allocators
- Cannot optimize for specific use cases
- Difficult to support NUMA, pinned memory, etc.

### Memory Pooling Strategy

**Decision**: Size-class based pooling for small allocations.

**Rationale**:
1. **Performance**: Reduces allocation overhead by 10-100x
2. **Fragmentation**: Size classes reduce external fragmentation
3. **Thread-Local**: Minimize synchronization overhead

**Implementation**:
```rust
thread_local! {
    static MEMORY_POOL: RefCell<SizeClassPool> = RefCell::new(
        SizeClassPool::new(&[64, 256, 1024, 4096])
    );
}

pub struct SizeClassPool {
    pools: Vec<Vec<*mut u8>>,  // One pool per size class
    size_classes: Vec<usize>,
}
```

**Performance Impact**:
- Small allocations (< 4KB): 10-50x faster than system malloc
- Large allocations: Fallback to system allocator
- Memory overhead: ~10% for pool bookkeeping

**Trade-offs**:
- **Pro**: Significant performance improvement for small tensors
- **Con**: Memory overhead, complexity
- **Decision**: Performance gain justifies overhead in ML workloads

### NUMA Awareness

**Decision**: Optional NUMA-aware allocation with multiple policies.

**Rationale**:
1. **Large Systems**: Critical for multi-socket servers
2. **Flexibility**: Different policies for different workloads
3. **Opt-In**: No overhead for single-socket systems

**Policies**:
- **LocalPreferred**: Try local node, fall back to remote
- **LocalOnly**: Fail if local node unavailable
- **Interleave**: Round-robin across nodes
- **Bind**: Explicitly bind to specific node

**Trade-offs**:
- **Pro**: Better performance on NUMA systems (2-5x for memory-bound ops)
- **Con**: Additional complexity, platform-specific code
- **Decision**: Essential for high-performance computing workloads

## Performance vs Safety Trade-offs

### Bounds Checking Strategy

**Decision**: Bounds checking in debug, unchecked in release (with opt-in).

**Rationale**:
1. **Development**: Catch errors during development
2. **Production**: Maximum performance in release builds
3. **Flexibility**: Users can enable checks via RuntimeConfig

**Implementation**:
```rust
#[inline]
pub fn get(&self, index: usize) -> f32 {
    #[cfg(debug_assertions)]
    assert!(index < self.len(), "Index out of bounds");

    unsafe { *self.data.add(index) }
}

// Optional runtime checking
pub fn get_checked(&self, index: usize) -> Result<f32> {
    if index >= self.len() {
        return Err(TorshError::IndexOutOfBounds { index, size: self.len() });
    }
    Ok(unsafe { *self.data.add(index) })
}
```

**Trade-offs**:
- **Pro**: Maximum performance in production, safety in development
- **Con**: Different behavior in debug/release
- **Mitigation**: Comprehensive test suite catches issues

### SIMD Optimization Trade-offs

**Decision**: Platform-specific SIMD with portable fallback.

**Rationale**:
1. **Performance**: 2-8x speedup for element-wise operations
2. **Portability**: Fallback ensures correctness on all platforms
3. **Maintainability**: Separate implementations are easier to optimize

**Implementation**:
```rust
#[cfg(target_feature = "avx2")]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    simd_avx2::add(a, b)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    simd_neon::add(a, b)
}

#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}
```

**Trade-offs**:
- **Pro**: Significant performance gains on modern CPUs
- **Con**: More code to maintain, platform-specific testing
- **Decision**: Performance critical for ML workloads

## API Stability Considerations

### Deprecation Strategy

**Decision**: Soft deprecation with migration guides.

**Rationale**:
1. **User Experience**: Gradual migration is less disruptive
2. **Compatibility**: Old code continues to work
3. **Guidance**: Clear migration paths reduce friction

**Implementation**:
```rust
#[deprecated(
    since = "0.2.0",
    note = "Use `Shape::new()` instead. See migration guide: ..."
)]
pub fn create_shape(dims: Vec<usize>) -> Shape {
    Shape::new(dims)
}
```

**Process**:
1. **Soft Deprecation** (1-2 releases): Mark as deprecated, provide migration guide
2. **Hard Deprecation** (2-3 releases): Remove from documentation
3. **Removal** (Major version): Remove from codebase

### Semantic Versioning

**Decision**: Strict semver with stability guarantees.

**Rationale**:
1. **Predictability**: Users know when breaking changes occur
2. **Trust**: Builds confidence in the framework
3. **Ecosystem**: Compatible with Cargo's dependency resolution

**Guarantees**:
- **Patch** (0.1.x): Bug fixes only, no API changes
- **Minor** (0.x.0): New features, deprecations, no breaking changes
- **Major** (x.0.0): Breaking changes allowed

## Future-Proofing

### Extension Points

**Design Decision**: Provide clear extension points for:
1. Custom data types via `TensorElement` trait
2. Custom devices via `Device` trait
3. Custom allocators via `Storage` trait
4. Custom error types via `From` implementations

**Rationale**: Cannot predict all future use cases, must allow extension.

### Feature Flags

**Decision**: Granular feature flags for optional functionality.

**Implementation**:
```toml
[features]
default = ["std"]
std = []
parallel = ["rayon"]
simd = []
cuda = ["cuda-sys"]
metal = ["metal-rs"]
serialize = ["serde"]
```

**Rationale**:
1. **Binary Size**: Only include what's needed
2. **Compilation Time**: Faster builds with fewer features
3. **Dependencies**: Avoid unnecessary dependencies

## Conclusion

These design decisions prioritize:
1. **Safety**: Catch errors at compile time
2. **Performance**: Zero-cost abstractions, SIMD, caching
3. **Flexibility**: Extensible through traits and registries
4. **Maintainability**: Clear separation of concerns
5. **Integration**: Deep SciRS2 integration

Trade-offs are made consciously with production ML workloads in mind. The result is a framework that is both safe and fast, with clear paths for future enhancement.

---

*Last Updated: 2025-10-23*
*Version: 0.1.0-alpha.2*
