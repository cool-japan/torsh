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

## Critical Dependencies and SciRS2 Policy

ToRSh **must** use SciRS2 as its foundation (see SCIRS2_INTEGRATION_POLICY.md):
- `scirs2-core` - Core scientific primitives (required) - **replaces direct rand and ndarray usage**
- `scirs2-optimize` - Base optimization interfaces (required)
- `optirs` - Advanced ML optimization algorithms
- Additional SciRS2 crates added based on compilation evidence

### FULL USE OF SciRS2-Core

ToRSh must make **FULL USE** of scirs2-core's extensive capabilities:

#### Core Array Operations (replaces ndarray)
```rust
// UNIFIED ACCESS: Complete ndarray functionality through scirs2-core
use scirs2_core::ndarray::*;  // Complete ndarray API including ALL macros

// Or selective imports for specific needs
use scirs2_core::ndarray::{
    Array, Array1, Array2, Array3, Array4, ArrayD,
    ArrayView, ArrayView1, ArrayView2, ArrayViewMut,
    Axis, Ix1, Ix2, IxDyn,
    array, arr1, arr2, s,     // ALL macros now available!
    concatenate, stack, azip  // Advanced operations
};

// Prelude for common usage
use scirs2_core::ndarray::prelude::*;

// Legacy compatibility (still works but discouraged)
use scirs2_autograd::ndarray::{Array, array};  // Autograd-specific usage
use scirs2_core::ndarray_ext::{stats, matrix}; // Extended utilities
```

#### Random Number Generation (replaces rand + rand_distr)
```rust
// UNIFIED ACCESS: Complete random functionality including all distributions
use scirs2_core::random::prelude::*;  // Common distributions & RNG

// All rand_distr distributions now available directly
use scirs2_core::random::{
    // Basic RNG
    Random, thread_rng, rng,

    // Continuous distributions
    Cauchy, ChiSquared, FisherF, LogNormal, StudentT, Weibull,
    RandBeta, InverseGaussian, Pareto, Pert, Triangular,

    // Discrete distributions
    Binomial, Poisson, Geometric, Hypergeometric, Zipf, Zeta,

    // Multivariate distributions
    RandDirichlet, UnitBall, UnitCircle, UnitDisc, UnitSphere,
};

// Enhanced unified interface with array sampling
use scirs2_core::random::distributions_unified::{
    UnifiedNormal, UnifiedBeta, UnifiedStudentT, UnifiedCauchy,
};

// Advanced features
use scirs2_core::random::{QuasiMonteCarloSequence, SecureRandom};
use scirs2_core::random::{ImportanceSampling, VarianceReduction};
```

#### Performance Optimization Features
```rust
// SIMD acceleration
use scirs2_core::simd::{SimdArray, SimdOps, auto_vectorize};
use scirs2_core::simd_ops::{simd_dot_product, simd_matrix_multiply};

// Parallel processing
use scirs2_core::parallel::{ParallelExecutor, ChunkStrategy, LoadBalancer};
use scirs2_core::parallel_ops::{par_chunks, par_join, par_scope};

// GPU acceleration
use scirs2_core::gpu::{GpuContext, GpuBuffer, GpuKernel, CudaBackend, MetalBackend};
use scirs2_core::tensor_cores::{TensorCore, MixedPrecision, AutoTuning};
```

#### Memory Management & Efficiency
```rust
// Memory-efficient operations
use scirs2_core::memory_efficient::{MemoryMappedArray, LazyArray, ChunkedArray};
use scirs2_core::memory_efficient::{ZeroCopyOps, AdaptiveChunking, DiskBackedArray};

// Memory management
use scirs2_core::memory::{BufferPool, GlobalBufferPool, ChunkProcessor};
use scirs2_core::memory::{LeakDetector, MemoryMetricsCollector};
```

#### Advanced Scientific Computing
```rust
// Complex numbers and numeric conversions
use scirs2_core::types::{ComplexOps, ComplexExt, NumericConversion};

// Scientific constants and units
use scirs2_core::constants::{math, physical, prefixes};
use scirs2_core::units::{UnitSystem, UnitRegistry, Dimension, convert};

// Validation and error handling
use scirs2_core::validation::{check_finite, check_in_bounds, ValidationSchema};
use scirs2_core::error::{CoreError, Result};
```

#### Production-Ready Features
```rust
// Performance profiling
use scirs2_core::profiling::{Profiler, profiling_memory_tracker};
use scirs2_core::benchmarking::{BenchmarkSuite, BenchmarkRunner};

// Metrics and monitoring
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use scirs2_core::observability::{audit, tracing};
```

### Mandatory Usage Guidelines

1. **NEVER** import `ndarray` directly - use `scirs2_core::ndarray` for complete unified functionality
2. **NEVER** import `rand` or `rand_distr` directly - use `scirs2_core::random` for all RNG and distributions
3. **ALWAYS** use `scirs2_core::ndarray::*` or `scirs2_core::ndarray::prelude::*` for array operations (includes ALL macros)
4. **ALWAYS** use `scirs2_core::random::prelude::*` for common distributions and RNG
5. **ALWAYS** use scirs2-core's SIMD operations for performance-critical code (if available)
6. **ALWAYS** use scirs2-core's GPU abstractions for hardware acceleration (if available)
7. **ALWAYS** use scirs2-core's memory management for large data operations (if available)
8. **ALWAYS** use scirs2-core's profiling and benchmarking tools (if available)

### ToRSh Module-Specific SciRS2 Usage

#### torsh-tensor
- Use `scirs2_core::ndarray::*` for complete array functionality (includes all macros)
- Use `scirs2_core::ndarray::prelude::*` for common operations
- Use `scirs2_core::simd_ops` for vectorized operations (check availability)
- Use `scirs2_core::parallel_ops` for parallel tensor operations (check availability)
- Legacy: `scirs2_autograd::ndarray` still works but discouraged

#### torsh-autograd
- Use `scirs2-autograd` for automatic differentiation with SafeVariableEnvironment
- Use `scirs2_core::ndarray::*` for array operations in autograd contexts
- Use `scirs2_core::memory_efficient` for gradient accumulation (check availability)
- Use `scirs2_core::random::prelude::*` for stochastic operations

#### torsh-nn
- Use `scirs2-neural` (via scirs2 features) as foundation for all layers
- Use `scirs2_core::jit` for optimized kernels where applicable

#### torsh-optim
- Use `scirs2-optimize` for base optimizer implementations
- Use `optirs` for advanced optimization algorithms
- Use `scirs2_core::random::prelude::*` for stochastic optimizers
- Use `scirs2_core::metrics` for optimization metrics

#### torsh-backend
- Use `scirs2_core::gpu` as foundation for GPU abstractions
- Use `scirs2_core::tensor_cores` for mixed-precision training
- Use `scirs2_core::array_protocol::GPUArray` for GPU array interface
- Use `scirs2_core::ndarray::*` for CPU backend arrays

#### torsh-data
- Use `scirs2` with datasets feature for data loading
- Use `scirs2_core::memory_efficient` for large dataset handling
- Use `scirs2_core::parallel::LoadBalancer` for parallel data loading
- Use `scirs2_core::random::prelude::*` for data augmentation

#### torsh-benches
- Use `scirs2_core::benchmarking` exclusively for all benchmarks
- Use `scirs2_core::profiling::Profiler` for detailed analysis
- Use `scirs2_core::metrics::MetricRegistry` for tracking
- Use `scirs2_core::ndarray::*` for test data generation

### Migration Checklist - Ensure Full SciRS2 Usage

When reviewing or writing ToRSh code, verify:

#### ✅ Arrays and Numerical Operations
- [ ] NO direct `use ndarray::{...}`
- [ ] NO direct `Array`, `Array1`, `Array2` from ndarray
- [ ] YES `use scirs2_core::ndarray::*` or `use scirs2_core::ndarray::prelude::*`
- [ ] YES `scirs2_core::ndarray::{array, s, azip}` macros available everywhere
- [ ] Legacy: `scirs2_autograd::ndarray` usage minimized

#### ✅ Random Number Generation and Distributions
- [ ] NO direct `use rand::{...}`
- [ ] NO direct `use rand_distr::{...}`
- [ ] YES `use scirs2_core::random::prelude::*` for common usage
- [ ] YES `use scirs2_core::random::{Cauchy, StudentT, Beta, ...}` for specific distributions
- [ ] YES `use scirs2_core::random::distributions_unified::*` for enhanced functionality

#### ✅ Performance Optimization
- [ ] YES use `scirs2_core::simd` for vectorized operations
- [ ] YES use `scirs2_core::parallel_ops` for parallelization
- [ ] YES use `scirs2_core::gpu` for GPU acceleration
- [ ] YES use `scirs2_core::memory_efficient` for large datasets

#### ✅ Production Features
- [ ] YES use `scirs2_core::error::{CoreError, Result}`
- [ ] YES use `scirs2_core::profiling` for performance analysis
- [ ] YES use `scirs2_core::metrics` for monitoring
- [ ] YES use `scirs2_core::benchmarking` for benchmarks

### Common Anti-Patterns to Avoid
```rust
// ❌ WRONG - Direct dependencies (POLICY VIOLATIONS)
use ndarray::{Array2, array, s};
use rand::{Rng, thread_rng};
use rand_distr::{Normal, Beta, StudentT};

// ❌ WRONG - Fragmented SciRS2 usage
use scirs2_autograd::ndarray::{Array2, array};  // Fragmented
use scirs2_core::ndarray_ext::{ArrayView};      // Missing macros
use ndarray::s;  // Still violating policy for s! macro

// ✅ CORRECT - Unified SciRS2 usage
use scirs2_core::ndarray::*;  // Complete unified access
// Or selective:
use scirs2_core::ndarray::{Array2, array, s, Axis};

use scirs2_core::random::prelude::*;  // Common distributions & RNG
// Or selective:
use scirs2_core::random::{thread_rng, Normal as RandNormal, RandBeta, StudentT};

// ✅ CORRECT - Enhanced unified interface
use scirs2_core::random::distributions_unified::{UnifiedNormal, UnifiedBeta};
```

## Testing Approach

The project uses Rust's built-in testing framework:
- Unit tests are located in `#[cfg(test)]` modules within source files
- Integration tests are in `tests/` directories within each crate
- Use `approx::assert_relative_eq!` for floating-point comparisons
- Tests should be deterministic - use seeded random number generation

## Important Notes

- Always run `make check` before committing to ensure code quality
- **CRITICAL**: Follow the SCIRS2_INTEGRATION_POLICY.md strictly - ToRSh MUST use SciRS2 as its scientific computing foundation
- **NEVER** use ndarray, rand, or rand_distr directly - ALWAYS use scirs2_core::ndarray and scirs2_core::random
- **UNIFIED ACCESS**: Use scirs2_core::ndarray::* and scirs2_core::random::prelude::* for complete functionality
- Make FULL USE of SciRS2's features to avoid reinventing the wheel
- The project is designed for ease of maintenance and readability
- The project follows Rust idioms and conventions
- Prefer editing existing files over creating new ones
- Use the latest available crate versions (as per the "Latest crates policy")
- When refactoring, keep single files under 2000 lines
- The CPU backend may have thread pool initialization warnings - these are expected
- Some crates (torsh-models, torsh-ffi) are temporarily disabled due to API compatibility issues

**Remember**: ToRSh is built on top of SciRS2, not as a standalone project. It must leverage the full power of the SciRS2 ecosystem to provide a PyTorch-compatible deep learning framework.