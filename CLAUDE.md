# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ToRSh Project Overview

ToRSh (Tensor Operations in Rust with Sharding) is a production-ready deep learning framework built in pure Rust. It leverages Rust's zero-cost abstractions, memory safety, and the scirs2 ecosystem to provide a PyTorch-compatible API with superior performance.

## Architecture

The project uses a modular workspace structure with specialized crates:

- **torsh-core**: Core types (Device, DType, Shape, Storage)
- **torsh-tensor**: Tensor implementation with strided storage and operations
- **torsh-autograd**: Automatic differentiation engine with gradient computation
- **torsh-nn**: Neural network modules (layers, activations, containers) - powered by scirs2-neural
- **torsh-optim**: Optimization algorithms (SGD, Adam, AdamW, AdaGrad, RMSprop)
- **torsh-data**: Data loading framework with parallel processing
- **torsh-backends**: Backend trait definitions
- **torsh-backend-cpu**: CPU backend with SIMD optimizations
- **torsh-backend-cuda**: CUDA GPU backend (partially implemented)
- **torsh-benches**: Benchmarking suite

Key dependencies:
- scirs2 (v0.1.0-beta.2): Scientific computing primitives with neural network support
- scirs2-optimize (v0.1.0-beta.2): Base optimization interfaces
- optirs (v0.1.0-beta.1): Advanced ML optimization algorithms
- numrs2 (v0.1.0-alpha.5): Numerical computing library
- polars (v0.48): For data manipulation

## Common Development Commands

### Building
```bash
# Build all packages
make build

# Build with release optimizations
make build-release

# Clean build artifacts
make clean
```

### Testing
```bash
# Run all tests (including slower backend tests)
make test

# Run tests quickly (excluding backend-cpu tests)
make test-fast

# Run tests for a specific crate
cargo test --package torsh-nn

# Run a specific test
cargo test --package torsh-nn test_name

# Run tests with output
cargo test -- --nocapture
```

### Code Quality
```bash
# Format code
make format

# Run clippy lints
make lint

# Run format + lint + test-fast in sequence
make check

# Audit dependencies for security issues
make audit
```

### Documentation
```bash
# Build and open documentation
make docs

# Build docs without opening
make docs-build
```

### Running Examples
```bash
# Run basic examples
make examples

# Run specific example
cargo run --example linear_regression
cargo run --example neural_network_training
cargo run --example advanced_training
```

### Benchmarking
```bash
# Run benchmarks
make bench

# Run specific benchmark
cargo bench --package torsh-benches <benchmark_name>
```

## Key Implementation Details

### Tensor Operations
All tensor operations are implemented through the scirs2 backend. The `torsh-tensor` crate provides a PyTorch-compatible wrapper API around scirs2's tensor functionality.

### Automatic Differentiation
The autograd system tracks operations through a computational graph. Key types:
- `GradFn`: Represents a differentiable operation
- `AutogradContext`: Manages the computation graph
- Tensors with `requires_grad=true` participate in gradient computation

### Neural Network Modules
All modules implement the `Module` trait which requires:
- `forward()`: Forward pass computation
- `parameters()`: Return trainable parameters
- `train()/eval()`: Switch between training and evaluation modes

### Backend System
The backend trait abstraction allows swapping between different compute backends:
- CPU backend uses Rayon for parallelism and SIMD for vectorization
- CUDA backend (in development) integrates with scirs2's GPU support

## 🚨 Critical: SciRS2 POLICY Compliance

ToRSh **must** follow the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) strictly (see [SCIRS2_INTEGRATION_POLICY.md](SCIRS2_INTEGRATION_POLICY.md)):

### **Mandatory Layered Architecture**
- **ONLY `scirs2-core` may use external dependencies directly** (rand, ndarray, num-traits, etc.)
- **ALL ToRSh crates MUST use `scirs2-core` abstractions** instead of direct external imports
- **NO direct imports** of `rand`, `rand_distr`, `ndarray`, `num_traits`, `num_complex` in ToRSh code
- This ensures: Consistent APIs, centralized version control, type safety, and maintainability

### UNIFIED ACCESS Through scirs2-core (v0.1.0-RC.1+)

ToRSh must use **UNIFIED scirs2-core abstractions** for all external functionality:

#### 1. Array Operations - UNIFIED ndarray Module (v0.1.0-RC.1+)

```rust
// ✅ PREFERRED: Complete unified ndarray access
use scirs2_core::ndarray::*;  // ALL ndarray functionality including macros
// OR selective:
use scirs2_core::ndarray::{
    // Core types
    Array, Array1, Array2, Array3, Array4, ArrayD,
    ArrayView, ArrayView1, ArrayView2, ArrayViewMut,

    // Essential macros - NOW AVAILABLE!
    array, arr1, arr2, s, azip,  // ALL macros work!

    // Common operations
    Axis, Ix1, Ix2, IxDyn,
    concatenate, stack,
};

// Prelude for common usage
use scirs2_core::ndarray::prelude::*;

// ⚠️ LEGACY (Still works but discouraged)
use scirs2_autograd::ndarray::{Array, array};  // Fragmented
use scirs2_core::ndarray_ext::{stats, matrix};  // Fragmented

// ❌ WRONG: Direct ndarray import
use ndarray::{Array, array, s};  // POLICY VIOLATION
```

#### 2. Random Number Generation - UNIFIED random Module (v0.1.0-RC.1+)

```rust
// ✅ PREFERRED: Complete random functionality
use scirs2_core::random::*;  // ALL rand + rand_distr functionality
// OR selective:
use scirs2_core::random::{
    // Basic RNG
    thread_rng, seeded_rng, CoreRandom,

    // Common distributions (directly available)
    Normal, Uniform, Exp, Gamma,
    RandBeta,  // Beta distribution (renamed to avoid conflict)

    // Advanced distributions
    Cauchy, ChiSquared, FisherF, LogNormal, StudentT, Weibull,
    Binomial, Poisson, Geometric,
};

// Prelude for common distributions
use scirs2_core::random::prelude::*;

// Advanced features (when available)
use scirs2_core::random::{
    qmc::{SobolGenerator, HaltonGenerator, LatinHypercubeSampler},
    variance_reduction::{ImportanceSampling, AntitheticSampling},
    secure::SecureRandom,
};

// ❌ WRONG: Direct rand imports
use rand::{thread_rng, Rng};           // POLICY VIOLATION
use rand_distr::{Normal, Beta};        // POLICY VIOLATION
```

#### 3. Numerical Traits - UNIFIED numeric Module

```rust
// ✅ REQUIRED: Use scirs2-core numeric abstractions
use scirs2_core::numeric::*;  // num-traits, num-complex, num-integer
// OR selective:
use scirs2_core::numeric::{Float, Zero, One, NumCast, ToPrimitive};

// Complex numbers
use scirs2_core::{Complex, Complex32, Complex64};

// ❌ WRONG: Direct num-traits imports
use num_traits::{Float, Zero};        // POLICY VIOLATION
use num_complex::Complex;              // POLICY VIOLATION
```

#### 4. Performance Optimization - MANDATORY scirs2-core Usage

```rust
// ✅ REQUIRED: SIMD acceleration through scirs2-core
use scirs2_core::simd_ops::SimdUnifiedOps;  // Unified SIMD operations
let result = f32::simd_add(&a.view(), &b.view());
let dot = f64::simd_dot(&x.view(), &y.view());

// ❌ FORBIDDEN: Direct SIMD in ToRSh modules
// use wide::f32x8;  // POLICY VIOLATION

// ✅ REQUIRED: Parallel processing through scirs2-core
use scirs2_core::parallel_ops::*;
let results: Vec<_> = (0..1000).into_par_iter().map(|x| x * x).collect();

// ❌ FORBIDDEN: Direct Rayon in ToRSh modules
// use rayon::prelude::*;  // POLICY VIOLATION

// ✅ REQUIRED: GPU operations through scirs2-core
#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuDevice, GpuKernel, GpuContext};

// ❌ FORBIDDEN: Direct CUDA/Metal calls
// use cuda_sys::*;  // POLICY VIOLATION
```

#### 5. Memory Management & Utilities

```rust
// Memory-efficient operations (when available)
#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{
    MemoryMappedArray, LazyArray, ChunkedArray,
    chunk_wise_op, create_mmap,
};

// Memory management (when available)
#[cfg(feature = "memory_management")]
use scirs2_core::memory::{
    BufferPool, GlobalBufferPool, ChunkProcessor,
    global_buffer_pool,
};

// Validation and error handling
use scirs2_core::validation::{check_positive, check_finite, checkarray_finite};
use scirs2_core::error::{CoreError, CoreResult};

// Scientific constants
use scirs2_core::constants::{math, physical};
```

#### 6. Production Features (when available)

```rust
// Performance profiling
#[cfg(feature = "profiling")]
use scirs2_core::profiling::Profiler;

// Benchmarking
#[cfg(feature = "benchmarking")]
use scirs2_core::benchmarking::{BenchmarkSuite, BenchmarkRunner};

// Metrics and monitoring
use scirs2_core::metrics::{
    global_metrics_registry, Counter, Gauge, Histogram,
};
```

### Mandatory SciRS2 POLICY Guidelines (CRITICAL)

#### ✅ REQUIRED Practices

1. **UNIFIED ndarray Access (v0.1.0-RC.1+)**
   - Use `scirs2_core::ndarray::*` for complete functionality (ALL macros included)
   - Single import point eliminates confusion and ensures POLICY compliance

2. **UNIFIED random Access (v0.1.0-RC.1+)**
   - Use `scirs2_core::random::*` for ALL RNG and distributions
   - Complete rand + rand_distr functionality through one module

3. **UNIFIED numeric Access**
   - Use `scirs2_core::numeric::*` for all numerical traits
   - Never import num-traits, num-complex directly

4. **Performance Through scirs2-core**
   - SIMD: `scirs2_core::simd_ops::SimdUnifiedOps` (MANDATORY)
   - Parallel: `scirs2_core::parallel_ops::*` (MANDATORY)
   - GPU: `scirs2_core::gpu` (MANDATORY when using GPU)

5. **Cargo.toml POLICY Compliance**
   ```toml
   # ✅ CORRECT: ToRSh module Cargo.toml
   [dependencies]
   scirs2-core = { workspace = true, features = ["ndarray", "random", "parallel"] }
   scirs2-autograd = { workspace = true }
   scirs2-neural = { workspace = true }

   # ❌ FORBIDDEN: Direct external dependencies
   # ndarray = { workspace = true }  # POLICY VIOLATION
   # rand = { workspace = true }      # POLICY VIOLATION
   # rayon = { workspace = true }     # POLICY VIOLATION
   ```

#### ❌ PROHIBITED Practices (POLICY VIOLATIONS)

```rust
// ❌ FORBIDDEN: Direct external imports
use ndarray::{Array, array, s};            // POLICY VIOLATION
use rand::{thread_rng, Rng};               // POLICY VIOLATION
use rand_distr::{Normal, Beta, StudentT};  // POLICY VIOLATION
use num_traits::{Float, Zero};             // POLICY VIOLATION
use rayon::prelude::*;                     // POLICY VIOLATION
use wide::f32x8;                           // POLICY VIOLATION
```

### ToRSh Module-Specific SciRS2 POLICY Compliance

#### torsh-tensor (Array Operations Core)
```rust
// ✅ REQUIRED: Unified ndarray access
use scirs2_core::ndarray::*;  // Complete functionality including macros

// ✅ REQUIRED: Performance operations
#[cfg(feature = "simd")]
use scirs2_core::simd_ops::SimdUnifiedOps;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

// ❌ FORBIDDEN
// use ndarray::{Array, array};  // POLICY VIOLATION
```

#### torsh-autograd (Automatic Differentiation)
```rust
// ✅ REQUIRED: Autograd from scirs2
use scirs2_autograd::*;  // SafeVariableEnvironment, Variable, etc.

// ✅ REQUIRED: Arrays through scirs2-core
use scirs2_core::ndarray::*;

// ✅ REQUIRED: Random for stochastic operations
use scirs2_core::random::*;

// Memory-efficient gradient accumulation (when available)
#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::chunk_wise_op;
```

#### torsh-nn (Neural Networks)
```rust
// ✅ REQUIRED: Neural network foundation
use scirs2_neural::*;  // Via scirs2 features

// ✅ REQUIRED: Arrays and random
use scirs2_core::ndarray::*;
use scirs2_core::random::*;

// JIT compilation (when available)
#[cfg(feature = "jit")]
use scirs2_core::jit::{JitCompiler, JitKernel};
```

#### torsh-optim (Optimization)
```rust
// ✅ REQUIRED: Base optimizers
use scirs2_optimize::*;

// ✅ REQUIRED: Advanced optimizers
use optirs::*;

// ✅ REQUIRED: Random for stochastic optimizers
use scirs2_core::random::*;

// ✅ REQUIRED: Metrics tracking
use scirs2_core::metrics::{global_metrics_registry, Counter, Histogram};
```

#### torsh-backend (Compute Backends)
```rust
// ✅ REQUIRED: GPU abstractions
#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuDevice, GpuKernel};

// ✅ REQUIRED: Tensor cores (when available)
#[cfg(feature = "tensor_cores")]
use scirs2_core::tensor_cores::{TensorCore, MixedPrecision};

// ✅ REQUIRED: Arrays for CPU backend
use scirs2_core::ndarray::*;

// ✅ REQUIRED: Parallel for CPU parallelism
#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;
```

#### torsh-data (Data Loading)
```rust
// ✅ REQUIRED: Dataset utilities
use scirs2_datasets::*;  // Via scirs2 features

// ✅ REQUIRED: Memory-efficient loading
#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{ChunkedArray, create_mmap};

// ✅ REQUIRED: Parallel data loading
#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

// ✅ REQUIRED: Data augmentation
use scirs2_core::random::*;
```

#### torsh-benches (Benchmarking)
```rust
// ✅ REQUIRED: Benchmarking framework
#[cfg(feature = "benchmarking")]
use scirs2_core::benchmarking::{BenchmarkSuite, BenchmarkRunner};

// ✅ REQUIRED: Profiling
#[cfg(feature = "profiling")]
use scirs2_core::profiling::Profiler;

// ✅ REQUIRED: Metrics
use scirs2_core::metrics::global_metrics_registry;

// ✅ REQUIRED: Test data generation
use scirs2_core::ndarray::*;
use scirs2_core::random::*;
```

### SciRS2 POLICY Compliance Checklist (MANDATORY)

When reviewing or writing ToRSh code, **ALWAYS verify**:

#### ✅ UNIFIED Arrays (v0.1.0-RC.1+)
- [ ] **NO** direct `use ndarray::{...}` (POLICY VIOLATION)
- [ ] **NO** direct `Array`, `Array1`, `Array2` from ndarray (POLICY VIOLATION)
- [ ] **YES** `use scirs2_core::ndarray::*` for complete functionality (ALL macros)
- [ ] **YES** `scirs2_core::ndarray::{array, s, azip}` macros work everywhere
- [ ] **LEGACY REMOVAL**: Minimize `scirs2_autograd::ndarray` usage (deprecated pattern)

#### ✅ UNIFIED Random (v0.1.0-RC.1+)
- [ ] **NO** direct `use rand::{...}` (POLICY VIOLATION)
- [ ] **NO** direct `use rand_distr::{...}` (POLICY VIOLATION)
- [ ] **YES** `use scirs2_core::random::*` for complete functionality
- [ ] **YES** Common distributions: `Normal`, `Uniform`, `RandBeta`, `StudentT`, `Cauchy`
- [ ] **YES** `use scirs2_core::random::prelude::*` for common patterns

#### ✅ UNIFIED Numerical Traits
- [ ] **NO** direct `use num_traits::{...}` (POLICY VIOLATION)
- [ ] **NO** direct `use num_complex::{...}` (POLICY VIOLATION)
- [ ] **YES** `use scirs2_core::numeric::*` for all numerical traits
- [ ] **YES** `use scirs2_core::{Complex, Complex32, Complex64}` for complex numbers

#### ✅ Performance Through scirs2-core (MANDATORY)
- [ ] **YES** `scirs2_core::simd_ops::SimdUnifiedOps` for SIMD (NO direct wide/packed_simd)
- [ ] **YES** `scirs2_core::parallel_ops::*` for parallel (NO direct rayon)
- [ ] **YES** `scirs2_core::gpu` for GPU (NO direct CUDA/Metal calls)
- [ ] **YES** `scirs2_core::memory_efficient` for large datasets

#### ✅ Cargo.toml POLICY Compliance
- [ ] **NO** `ndarray = { workspace = true }` in dependencies (POLICY VIOLATION)
- [ ] **NO** `rand = { workspace = true }` in dependencies (POLICY VIOLATION)
- [ ] **NO** `rayon = { workspace = true }` in dependencies (POLICY VIOLATION)
- [ ] **YES** `scirs2-core = { workspace = true, features = [...] }`
- [ ] **YES** Only SciRS2 crates in dependencies (scirs2-*, optirs)

#### ✅ Production Features (when available)
- [ ] **YES** `scirs2_core::error::{CoreError, CoreResult}` for errors
- [ ] **YES** `scirs2_core::validation` for input validation
- [ ] **YES** `scirs2_core::profiling::Profiler` for performance analysis
- [ ] **YES** `scirs2_core::metrics` for monitoring
- [ ] **YES** `scirs2_core::benchmarking` for benchmarks

### Common Anti-Patterns to Avoid (POLICY VIOLATIONS)

```rust
// ❌ WRONG - Direct external dependencies (CRITICAL POLICY VIOLATIONS)
use ndarray::{Array2, array, s};              // POLICY VIOLATION
use rand::{Rng, thread_rng};                  // POLICY VIOLATION
use rand_distr::{Normal, Beta, StudentT};     // POLICY VIOLATION
use num_traits::{Float, Zero};                // POLICY VIOLATION
use rayon::prelude::*;                        // POLICY VIOLATION

// ❌ WRONG - Fragmented SciRS2 usage (Deprecated Patterns)
use scirs2_autograd::ndarray::{Array2, array};  // Old fragmented pattern
use scirs2_core::ndarray_ext::{ArrayView};      // Old fragmented pattern
use ndarray::s;  // Still violating policy

// ✅ CORRECT - UNIFIED SciRS2 Access (v0.1.0-RC.1+)
use scirs2_core::ndarray::*;  // Complete unified access (ALL macros)
// OR selective:
use scirs2_core::ndarray::{Array2, array, s, Axis};

use scirs2_core::random::*;  // Complete unified access (ALL distributions)
// OR selective:
use scirs2_core::random::{thread_rng, Normal, RandBeta, StudentT};

use scirs2_core::numeric::*;  // Complete numerical traits
// OR selective:
use scirs2_core::numeric::{Float, Zero, One};
```

### Quick Migration Examples

#### Example 1: Array Operations

```rust
// ❌ OLD (Policy-Violating)
use ndarray::{Array, Array2, array, s};
let matrix = array![[1, 2], [3, 4]];
let slice = matrix.slice(s![.., 0]);

// ✅ NEW (Policy-Compliant)
use scirs2_core::ndarray::{Array, Array2, array, s};
let matrix = array![[1, 2], [3, 4]];  // array! macro works
let slice = matrix.slice(s![.., 0]);  // s! macro works
```

#### Example 2: Random Sampling

```rust
// ❌ OLD (Policy-Violating)
use rand::thread_rng;
use rand_distr::{Normal, Beta};
let mut rng = thread_rng();
let normal = Normal::new(0.0, 1.0)?;
let beta = Beta::new(2.0, 5.0)?;

// ✅ NEW (Policy-Compliant)
use scirs2_core::random::{thread_rng, Normal, RandBeta};
let mut rng = thread_rng();
let normal = Normal::new(0.0, 1.0)?;
let beta = RandBeta::new(2.0, 5.0)?;  // Note: RandBeta (renamed)
```

#### Example 3: Tensor Initialization

```rust
// ❌ OLD (Policy-Violating)
use ndarray::{Array2, array};
use rand::{thread_rng, Rng};
use rand_distr::Normal;

fn xavier_init(shape: (usize, usize)) -> Array2<f32> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32)
}

// ✅ NEW (Policy-Compliant)
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{thread_rng, Normal};

fn xavier_init(shape: (usize, usize)) -> Array2<f32> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32)
}
```

## Testing Approach

The project uses Rust's built-in testing framework:
- Unit tests are located in `#[cfg(test)]` modules within source files
- Integration tests are in `tests/` directories within each crate
- Use `approx::assert_relative_eq!` for floating-point comparisons
- Tests should be deterministic - use seeded random number generation

## Important Notes

### 🚨 CRITICAL: SciRS2 POLICY Enforcement

- **MANDATORY**: Follow the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) strictly
- **MANDATORY**: Follow the [SCIRS2_INTEGRATION_POLICY.md](SCIRS2_INTEGRATION_POLICY.md) for ToRSh-specific guidance
- **ONLY `scirs2-core` may use external dependencies directly** - ALL ToRSh crates use scirs2-core abstractions
- **ZERO TOLERANCE** for direct external imports (ndarray, rand, num-traits, rayon, etc.) in ToRSh code

### ✅ Required Practices (POLICY Compliance)

1. **UNIFIED ndarray Access (v0.1.0-RC.1+)**
   - Use `scirs2_core::ndarray::*` for ALL array operations (includes ALL macros)
   - NEVER import ndarray directly

2. **UNIFIED random Access (v0.1.0-RC.1+)**
   - Use `scirs2_core::random::*` for ALL RNG and distributions
   - NEVER import rand or rand_distr directly

3. **UNIFIED numeric Access**
   - Use `scirs2_core::numeric::*` for ALL numerical traits
   - NEVER import num-traits or num-complex directly

4. **Performance Through scirs2-core**
   - SIMD: `scirs2_core::simd_ops::SimdUnifiedOps` (MANDATORY)
   - Parallel: `scirs2_core::parallel_ops::*` (MANDATORY)
   - GPU: `scirs2_core::gpu` (MANDATORY when using GPU)

### 📋 Development Workflow

- Always run `make check` before committing (format + lint + test-fast)
- Make FULL USE of SciRS2's extensive features to avoid reinventing the wheel
- Prefer editing existing files over creating new ones
- Use the latest available crate versions (as per "Latest crates policy")
- When refactoring, keep single files under 2000 lines (use splitrs if needed)

### ⚠️ Known Issues

- CPU backend may have thread pool initialization warnings (expected behavior)
- Some crates temporarily disabled due to API compatibility issues (see SCIRS2_INTEGRATION_POLICY.md)

### 🎯 Project Philosophy

**ToRSh is built ON TOP OF SciRS2, not as a standalone project.** It leverages the complete SciRS2 ecosystem to provide a PyTorch-compatible deep learning framework while maintaining strict POLICY compliance for:
- Consistent APIs across all modules
- Centralized dependency management
- Type safety and maintainability
- Superior performance through unified optimizations

**Remember**: ToRSh's success depends on proper SciRS2 integration. Follow the POLICY strictly!