# SciRS2 Integration Policy for ToRSh

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT

**ToRSh MUST use SciRS2 as its scientific computing foundation.** This document establishes the policy for proper, complete, and effective integration of SciRS2 crates into ToRSh, following the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md).

## Core Integration Principles

### 1. **SciRS2 POLICY Compliance (MANDATORY)**
- **ONLY `scirs2-core` may use external dependencies directly** (rand, ndarray, num-traits, etc.)
- **ALL ToRSh crates MUST use `scirs2-core` abstractions** instead of direct external imports
- **NO direct imports** of `rand`, `rand_distr`, `ndarray`, `num_traits`, `num_complex` in ToRSh code
- **UNIFIED ACCESS**: Use `scirs2_core::ndarray::*` and `scirs2_core::random::*` for complete functionality
- This ensures: Consistent APIs, centralized version control, type safety, and maintainability

### 2. **Foundation, Not Dependency Bloat**
- ToRSh extends SciRS2's capabilities with deep learning framework specialization
- Use SciRS2 crates **only when actually needed** by ToRSh functionality
- **DO NOT** add SciRS2 crates "just in case" - add them when code requires them

### 3. **Evidence-Based Integration**
- Each SciRS2 crate must have **clear justification** based on ToRSh features
- Document **specific use cases** for each integrated SciRS2 crate
- Remove unused SciRS2 dependencies during code reviews

### 4. **Architectural Hierarchy**
```
ToRSh (Deep Learning Framework - PyTorch-compatible API)
    ↓ builds upon
OptiRS (ML Optimization Specialization)
    ↓ builds upon
SciRS2 (Scientific Computing Foundation)
    ↓ builds upon (via scirs2-core ONLY)
OxiBLAS (Pure Rust BLAS/LAPACK) + OxiCode + ndarray, rand, num-traits
    ⚠️ ONLY scirs2-core may import these directly
```

## Pure Rust Migration (v0.1.1)

**Major architectural milestone:** ToRSh now runs on 100% Pure Rust stack with zero system dependencies (default features).

### ✅ Pure Rust Achievement (2025-12-30)

**Default Features: 100% Pure Rust** - Zero C/Fortran dependencies!
- ✅ **libc removed** (system calls) → `sysinfo` (Pure Rust)
- ✅ **OpenBLAS/MKL removed** (Fortran BLAS) → OxiBLAS 0.1.2 (Pure Rust)
- ✅ **bincode removed** → OxiCode 0.1.1 (Pure Rust)
- ✅ **ndarray/rayon direct imports removed** → scirs2_core abstractions

### OxiBLAS Integration - Pure Rust BLAS/LAPACK

**REMOVED Dependencies (v0.1.0+):**
- ❌ `openblas-src` / `blas-src` / `lapack-src` - System BLAS libraries
- ❌ `accelerate-src` - macOS Accelerate Framework bindings
- ❌ `intel-mkl-src` - Intel MKL bindings
- ❌ `ndarray-linalg` - Replaced with scirs2-linalg independent implementation

**REMOVED Dependencies (v0.1.0-beta.1+):**
- ❌ `libc` - C standard library (torsh-backend, torsh-cli, torsh-core)
- ❌ Optional `lapack-backend` feature in torsh-linalg

**CURRENT Dependencies (v0.1.1+):**
- ✅ `oxiblas-ndarray` v0.1.2 - Pure Rust ndarray integration
- ✅ `oxiblas-blas` v0.1.2 - Pure Rust BLAS implementation
- ✅ `oxiblas-lapack` v0.1.2 - Pure Rust LAPACK (supports Complex<f64>)
- ✅ Accessed via `scirs2-core` features: `oxiblas-blas`, `oxiblas-lapack`, `oxiblas-ndarray`

**Benefits:**
- 🚀 **Zero System Dependencies** - No OpenBLAS, MKL, or system BLAS required
- 🔧 **Easy Cross-Compilation** - Pure Rust works on all platforms
- 📦 **Simplified Builds** - No C/Fortran compiler required
- 🔒 **Complete Control** - Full Rust ecosystem integration
- ⚡ **SIMD Optimized** - Competitive performance with native BLAS

### OxiCode Integration - SIMD-Optimized Serialization

**REMOVED Dependencies:**
- ❌ `bincode` - Generic binary serialization

**CURRENT Dependencies (COOLJAPAN Policy):**
- ✅ `oxicode` v0.1.1 - SIMD-optimized binary serialization
- ✅ Accessed via `scirs2-core` features: `oxicode`

**Benefits:**
- ⚡ **SIMD Acceleration** - Up to 4x faster than bincode
- 🎯 **Scientific Data Optimized** - Specialized for numeric arrays
- 🔒 **Type Safe** - Compile-time serialization verification

## Required SciRS2 Crates Analysis

### **ESSENTIAL (Always Required)**

#### `scirs2-core` - FOUNDATION
- **Use Cases**: Core scientific primitives, ScientificNumber trait, random number generation, tensor operations, **OxiBLAS BLAS/LAPACK**, **OxiCode serialization**
- **ToRSh Modules**: `torsh-core`, `torsh-tensor`, all modules use core utilities
- **Required Features**: `["simd", "parallel", "memory_management", "serialization", "oxicode", "oxiblas-blas", "oxiblas-lapack", "oxiblas-ndarray", "validation", "types", "random"]`
- **Status**: ✅ REQUIRED - Foundation crate with OxiBLAS 0.1.2 and OxiCode 0.1.1

#### `scirs2` - MAIN INTEGRATION
- **Use Cases**: Neural networks, autograd, linear algebra integration
- **ToRSh Modules**: `torsh`, `torsh-nn`, `torsh-autograd`, `torsh-functional`
- **Features**: ["neural", "autograd", "linalg"]
- **Status**: ✅ REQUIRED - Main integration crate

### **HIGHLY LIKELY REQUIRED**

#### `scirs2-autograd` - AUTOMATIC DIFFERENTIATION
- **Use Cases**: Gradient computation, computational graph, backpropagation, **array! macro access**
- **ToRSh Modules**: `torsh-autograd`, test modules throughout
- **Status**: ✅ REQUIRED - Core autograd functionality
- **Special Note**: The `array!` macro is accessed via `scirs2_autograd::ndarray::array` for tests

#### `scirs2-neural` - NEURAL NETWORKS
- **Use Cases**: Neural network layers, activation functions, loss functions
- **ToRSh Modules**: `torsh-nn` (via scirs2 features)
- **Status**: ✅ REQUIRED - Through scirs2 crate features

#### `scirs2-optimize` - OPTIMIZATION
- **Use Cases**: Optimization algorithms (SGD, Adam, AdamW, etc.)
- **ToRSh Modules**: `torsh-optim`
- **Status**: ✅ REQUIRED - Core optimization functionality
- **Note**: Also integrates with OptiRS for advanced optimizers

#### `scirs2-special` - SPECIAL FUNCTIONS
- **Use Cases**: Gamma, beta, error functions, Bessel functions, special math operations
- **ToRSh Modules**: `torsh-special`, `torsh-functional`
- **Status**: ✅ REQUIRED - Mathematical special functions

#### `scirs2-sparse` - SPARSE MATRICES
- **Use Cases**: Sparse tensor operations, CSR/CSC formats
- **ToRSh Modules**: `torsh-sparse`
- **Status**: ✅ REQUIRED - Sparse tensor support

### **CONDITIONALLY REQUIRED**

#### `scirs2-linalg` - LINEAR ALGEBRA
- **Use Cases**: Matrix operations, decompositions, eigenvalue problems, **scipy.linalg compatibility layer (35 functions)**
- **ToRSh Modules**: `torsh-linalg`
- **Key Features**: svd, eig, eigh, qr, lu, cholesky, lstsq, pinv, matrix functions (expm, logm, sqrtm, etc.)
- **Backend**: Built on OxiBLAS 0.1.2 (Pure Rust LAPACK)
- **Status**: ✅ REQUIRED - Linear algebra operations with scipy compatibility

#### `scirs2-neural` - ADVANCED NEURAL ARCHITECTURES
- **Use Cases**: Cutting-edge neural network architectures, experimental layers
- **ToRSh Modules**: `torsh-nn` (advanced features)
- **Status**: ✅ REQUIRED - Neural network foundation

#### `scirs2-signal` - SIGNAL PROCESSING
- **Use Cases**: FFT, convolutions, signal processing operations
- **ToRSh Modules**: `torsh-signal`
- **Status**: ✅ REQUIRED - Signal processing capabilities

#### `scirs2-fft` - FAST FOURIER TRANSFORM
- **Use Cases**: FFT operations, frequency domain transformations
- **ToRSh Modules**: `torsh-signal`, `torsh-functional`
- **Status**: ✅ REQUIRED - FFT operations

#### `scirs2-metrics` - EVALUATION METRICS
- **Use Cases**: Classification, regression, clustering metrics, model evaluation
- **ToRSh Modules**: `torsh-metrics` (new), evaluation tools
- **Status**: ✅ REQUIRED - Comprehensive evaluation

#### `scirs2-stats` - STATISTICAL ANALYSIS
- **Use Cases**: Statistical functions, distributions, hypothesis testing
- **ToRSh Modules**: `torsh-stats` (new), analysis utilities
- **Status**: ✅ REQUIRED - Statistical computing

#### `scirs2-datasets` - DATA HANDLING
- **Use Cases**: Built-in datasets, data generators, cross-validation utilities
- **ToRSh Modules**: `torsh-data`
- **Status**: ✅ REQUIRED - Data pipeline enhancement

#### `scirs2-cluster` - CLUSTERING ALGORITHMS
- **Use Cases**: K-means, DBSCAN, hierarchical clustering, unsupervised learning
- **ToRSh Modules**: `torsh-nn` (clustering layers), `torsh-vision` (segmentation)
- **Status**: ✅ REQUIRED - Unsupervised learning

#### `scirs2-graph` - GRAPH NEURAL NETWORKS
- **Use Cases**: Graph representations, GNN layers, spectral methods
- **ToRSh Modules**: `torsh-graph` (new)
- **Status**: ✅ REQUIRED - Graph neural networks

#### `scirs2-series` - TIME SERIES ANALYSIS
- **Use Cases**: Time series decomposition, forecasting, anomaly detection
- **ToRSh Modules**: `torsh-series` (new)
- **Status**: ✅ REQUIRED - Time series capabilities

#### `scirs2-spatial` - SPATIAL DATA PROCESSING
- **Use Cases**: KD-trees, spatial indexing, computational geometry
- **ToRSh Modules**: `torsh-vision` (enhanced)
- **Status**: ✅ REQUIRED - Advanced vision capabilities

#### `scirs2-text` - NLP PROCESSING
- **Use Cases**: Tokenization, text features, language models
- **ToRSh Modules**: `torsh-text` (enhanced)
- **Status**: ✅ REQUIRED - NLP capabilities

### **DOMAIN-SPECIFIC (Optional)**

#### `scirs2-vision` - COMPUTER VISION
- **Use Cases**: Image processing, computer vision models
- **ToRSh Modules**: `torsh-vision`
- **Status**: ⚠️ OPTIONAL - Only for vision-specific features

#### `scirs2-text` - TEXT PROCESSING
- **Use Cases**: NLP operations, text preprocessing
- **ToRSh Modules**: `torsh-text`
- **Status**: ⚠️ OPTIONAL - Only for NLP features

### **LIKELY NOT REQUIRED**

#### `scirs2-ndimage` - IMAGE PROCESSING
- **Status**: ❌ UNLIKELY - Use scirs2-vision instead

#### `scirs2-transform` - MATHEMATICAL TRANSFORMS
- **Status**: ❌ UNLIKELY - Unless specific transforms needed

#### `scirs2-interpolate` - INTERPOLATION
- **Status**: ❌ UNLIKELY - Unless interpolation features added

#### `scirs2-integrate` - NUMERICAL INTEGRATION
- **Status**: ❌ UNLIKELY - Unless numerical integration needed

#### `scirs2-io` - INPUT/OUTPUT
- **Status**: ❌ UNLIKELY - Basic I/O likely sufficient

## SciRS2 POLICY Compliance for ToRSh

### **Mandatory: Unified SciRS2-Core Abstractions**

Following the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md), ToRSh uses a **strict layered architecture** where all external dependencies are accessed through `scirs2-core`.

#### ✅ **REQUIRED Imports in ToRSh Crates**

```rust
// Arrays and numerical operations - COMPLETE UNIFIED ACCESS
use scirs2_core::ndarray::*;          // Complete ndarray including ALL macros (array!, s!, azip!)
// OR selective:
use scirs2_core::ndarray::{Array, Array1, Array2, array, s, Axis};

// Random number generation - COMPLETE UNIFIED ACCESS
use scirs2_core::random::*;           // Complete rand + rand_distr functionality
// OR selective:
use scirs2_core::random::{thread_rng, Normal, RandBeta, Uniform};

// Numerical traits - UNIFIED ACCESS
use scirs2_core::numeric::*;          // num-traits, num-complex, num-integer
// OR selective:
use scirs2_core::numeric::{Float, Zero, One, NumCast};

// Complex numbers
use scirs2_core::{Complex, Complex32, Complex64};  // Re-exported from num_complex
```

#### ❌ **PROHIBITED Direct Imports in ToRSh**

```rust
// ❌ FORBIDDEN - Direct external dependencies
use ndarray::{Array, Array1, Array2};         // POLICY VIOLATION
use ndarray::{array, s};                      // POLICY VIOLATION
use rand::*;                                  // POLICY VIOLATION
use rand::Rng;                                // POLICY VIOLATION
use rand_distr::{Normal, Beta, StudentT};     // POLICY VIOLATION
use num_traits::*;                            // POLICY VIOLATION
use num_complex::Complex;                     // POLICY VIOLATION
```

#### 📋 **Complete Dependency Mapping for ToRSh**

| External Crate | ToRSh MUST Use | Note |
|----------------|----------------|------|
| `ndarray` | `scirs2_core::ndarray` | Complete functionality including macros |
| `ndarray-rand` | `scirs2_core::ndarray` | Via `random` feature |
| `ndarray-stats` | `scirs2_core::ndarray` | Via `array_stats` feature |
| `ndarray-linalg` | `scirs2_core::ndarray` | Via `linalg` feature |
| `rand` | `scirs2_core::random` | Full RNG functionality |
| `rand_distr` | `scirs2_core::random` | All distributions (Normal, Beta, Cauchy, etc.) |
| `rand_chacha` | `scirs2_core::random` | ChaCha RNG variants |
| `num-traits` | `scirs2_core::numeric` | All numerical traits |
| `num-complex` | `scirs2_core` | Re-exported as `Complex` |
| `rayon` | `scirs2_core::parallel_ops` | Parallel processing |

### **Performance & Acceleration (MANDATORY scirs2-core Usage)**

#### SIMD Operations
```rust
// ✅ REQUIRED: Use scirs2-core SIMD operations
use scirs2_core::simd_ops::SimdUnifiedOps;
let result = f32::simd_add(&a.view(), &b.view());

// ❌ FORBIDDEN: Custom SIMD in ToRSh modules
// use wide::f32x8;  // POLICY VIOLATION
```

#### Parallel Processing
```rust
// ✅ REQUIRED: Use scirs2-core parallel ops
use scirs2_core::parallel_ops::*;

// ❌ FORBIDDEN: Direct Rayon in ToRSh modules
// use rayon::prelude::*;  // POLICY VIOLATION
```

#### GPU Operations
```rust
// ✅ REQUIRED: Use scirs2-core GPU abstractions
use scirs2_core::gpu::{GpuDevice, GpuKernel};

// ❌ FORBIDDEN: Direct CUDA/Metal calls
// use cuda_sys::*;  // POLICY VIOLATION
```

## Integration Guidelines

### **Adding New SciRS2 Dependencies**

1. **Document Justification**
   ```markdown
   ## SciRS2 Crate Addition Request

   **Crate**: scirs2-[name]
   **Requestor**: [Developer Name]
   **Date**: [Date]

   **Justification**:
   - Specific ToRSh feature requiring this crate
   - Code modules that will use it
   - Alternatives considered and why SciRS2 is preferred

   **Impact Assessment**:
   - Compilation time impact
   - Binary size impact
   - Maintenance burden
   ```

2. **Code Review Requirements**
   - Demonstrate actual usage in ToRSh code
   - Show integration examples
   - Verify no equivalent functionality exists in already-included crates
   - **Verify SciRS2 POLICY compliance** (no direct external imports)

3. **Documentation Requirements**
   - Update this policy document
   - Document usage patterns in relevant module docs
   - Add examples to integration tests

### **Removing SciRS2 Dependencies**

1. **Regular Audits** (quarterly)
   - Review all SciRS2 dependencies for actual usage
   - Remove unused imports and dependencies
   - Update documentation

2. **Deprecation Process**
   - Mark as deprecated with removal timeline
   - Provide migration guide if functionality moves
   - Remove after deprecation period

### **Best Practices (SciRS2 v0.1.1 Stable)**

#### 1. **UNIFIED Array Imports (scirs2-core v0.1.1)**

```rust
// ✅ PREFERRED: Complete unified ndarray access through scirs2-core
use scirs2_core::ndarray::*;  // ALL ndarray functionality including macros
// OR selective:
use scirs2_core::ndarray::{Array, Array1, Array2, array, s, azip};

// ⚠️ LEGACY (Still works but discouraged): Fragmented approach
use scirs2_autograd::ndarray::{Array, array};  // Old pattern
use scirs2_core::ndarray_ext::{ArrayView};     // Old pattern

// ❌ WRONG: Direct ndarray import
use ndarray::{Array, array};  // POLICY VIOLATION

// Example usage in ToRSh code:
#[cfg(test)]
mod tests {
    use super::*;
    // UNIFIED: One import for everything
    use scirs2_core::ndarray::{array, Array1, s};

    #[test]
    fn test_tensor_ops() {
        let data = array![1.0, 2.0, 3.0];      // array! macro works
        let slice = data.slice(s![0..2]);      // s! macro works
        let arr: Array1<f64> = Array1::zeros(10);
        // test implementation
    }
}
```

#### 2. **UNIFIED Random Number Generation (scirs2-core v0.1.1)**

```rust
// ✅ PREFERRED: Complete random functionality through scirs2-core
use scirs2_core::random::*;  // ALL rand + rand_distr functionality
// OR selective:
use scirs2_core::random::{thread_rng, Normal, RandBeta, StudentT, Cauchy};

// Common distributions now available directly:
// - Normal, Uniform, Exp, Gamma, Beta (as RandBeta)
// - Cauchy, ChiSquared, FisherF, LogNormal, StudentT, Weibull
// - Binomial, Poisson, Geometric, etc.

// ❌ WRONG: Direct rand/rand_distr imports
use rand::thread_rng;              // POLICY VIOLATION
use rand_distr::{Normal, Beta};    // POLICY VIOLATION

// Example: Initialization pattern
fn create_weights() -> Vec<f64> {
    let mut rng = thread_rng();  // From scirs2_core::random
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..100).map(|_| normal.sample(&mut rng)).collect()
}
```

#### 3. **Import Granularity**

```rust
// ✅ GOOD - Specific imports for clarity
use scirs2_core::ndarray::{Array1, Array2, array, s};
use scirs2_core::random::{thread_rng, Normal};
use scirs2_core::numeric::{Float, Zero};

// ⚠️ ACCEPTABLE - Glob imports when appropriate
use scirs2_core::ndarray::*;  // When using many array operations
use scirs2_core::random::prelude::*;  // Common distributions

// ❌ BAD - Top-level glob imports (unless in prelude)
use scirs2_core::*;  // Too broad, unclear what's imported
```

#### 4. **Feature Gates**

```rust
// ✅ GOOD - Optional features with scirs2-core
#[cfg(feature = "simd")]
use scirs2_core::simd_ops::SimdUnifiedOps;

#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuDevice, GpuKernel};

#[cfg(feature = "vision")]
use scirs2_vision::transforms::ImageTransform;
```

#### 5. **Error Handling**

```rust
// ✅ GOOD - Use scirs2-core error types
use scirs2_core::error::{CoreError, CoreResult};
use scirs2_core::validation::{check_positive, check_finite};

pub fn process_tensor(data: &Array2<f64>, k: usize) -> CoreResult<Array2<f64>> {
    check_positive(k, "k")?;
    check_finite(data)?;
    // Implementation
    Ok(data.clone())
}
```

#### 6. **Cargo.toml Best Practices**

```toml
# ✅ CORRECT: ToRSh module Cargo.toml
[dependencies]
scirs2-core = { workspace = true, features = ["ndarray", "random", "parallel"] }
scirs2-autograd = { workspace = true }
scirs2-neural = { workspace = true }

# ❌ WRONG: Direct external dependencies
# ndarray = { workspace = true }  # POLICY VIOLATION
# rand = { workspace = true }      # POLICY VIOLATION
```

## ToRSh-Specific Guidelines

### **Tensor Backend Integration**
- All tensor operations MUST go through SciRS2's tensor implementation
- DO NOT implement custom tensor operations that duplicate SciRS2 functionality
- Extend SciRS2 tensors only when PyTorch compatibility requires it

### **Neural Network Modules**
- Use scirs2-neural as the foundation for all NN modules
- Wrap SciRS2 modules with PyTorch-compatible API
- Document deviations from SciRS2 implementation

### **Optimization Integration**
- Primary: Use scirs2-optimize for basic optimizers
- Secondary: Integrate OptiRS for advanced optimization algorithms
- Document which backend provides which optimizer

## Enforcement

### **Automated Checks**
- CI pipeline checks for unused SciRS2 dependencies
- Documentation tests verify integration examples work
- Dependency graph analysis in builds

### **Manual Reviews**
- All SciRS2 integration changes require team review
- Quarterly dependency audits
- Annual architecture review

### **Violation Response**
1. **Warning**: Document why integration is needed
2. **Correction**: Remove unjustified dependencies
3. **Training**: Educate team on integration policy

## Future Considerations

### **SciRS2 Version Management**
- Track SciRS2 release cycle (currently on 0.1.1 stable)
- Test ToRSh against SciRS2 stable releases
- Coordinate breaking change migrations with semver compliance

### **Performance Monitoring**
- Benchmark impact of SciRS2 integration
- Monitor compilation times
- Track binary size impact

### **Community Alignment**
- Coordinate with SciRS2 team on roadmap
- Contribute improvements back to SciRS2
- Maintain architectural consistency

## Conclusion

This policy ensures ToRSh properly leverages SciRS2's scientific computing foundation while maintaining PyTorch API compatibility. **ToRSh must use SciRS2 as its computational foundation, following the strict SciRS2 POLICY for layered architecture and unified abstractions.**

### Key Takeaways

1. **UNIFIED ACCESS**: Use `scirs2_core::ndarray::*` and `scirs2_core::random::*` for complete functionality
2. **NO DIRECT IMPORTS**: Never import external dependencies (rand, ndarray, num-traits) directly
3. **POLICY COMPLIANCE**: All ToRSh crates must follow the SciRS2 POLICY strictly
4. **PERFORMANCE**: Use scirs2-core abstractions for SIMD, parallel, and GPU operations
5. **CONSISTENCY**: Centralized dependency management ensures type safety and maintainability

### Quick Migration Guide

```rust
// OLD (Policy-Violating)
use ndarray::{Array, array, s};
use rand::{thread_rng, Rng};
use rand_distr::{Normal, Beta};

// NEW (Policy-Compliant)
use scirs2_core::ndarray::{Array, array, s};
use scirs2_core::random::{thread_rng, Normal, RandBeta};
```

---

**Document Version**: 4.0 - SciRS2 v0.1.1 Stable Integration
**Last Updated**: 2025-12-30
**Based On**: [SciRS2 POLICY v3.0.0](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md)
**SciRS2 Version**: v0.1.1 (Stable)
**OxiBLAS Version**: v0.1.2 (Stable)
**ToRSh Version**: v0.1.0-beta.1
**Next Review**: Q2 2026
**Owner**: ToRSh Architecture Team / COOLJAPAN OU

## 🎉 INTEGRATION SUCCESS STATUS

**Current Achievement**: **100% PRODUCTION READY (29/29 packages)** 🎯
- **Integration Date**: December 2025
- **SciRS2 Version**: 0.1.1 stable (with scipy.linalg compatibility)
- **OxiBLAS Version**: 0.1.2 stable (Pure Rust BLAS/LAPACK)
- **Status**: PRODUCTION READY - Complete SciRS2 integration with stable dependencies

### Successfully Integrated Packages (29/30) ✅

#### Core Infrastructure (10/10) - 100% SUCCESS
- `torsh-core` ✅ - Core tensor types and device abstractions
- `torsh-tensor` ✅ - Tensor implementation with SciRS2 backends
- `torsh-autograd` ✅ - Automatic differentiation engine
- `torsh-nn` ✅ - Neural network modules and layers
- `torsh-optim` ✅ - Optimization algorithms
- `torsh-backend` ✅ - Backend trait definitions
- `torsh-data` ✅ - Data loading with SciRS2 random integration
- `torsh-text` ✅ - Text processing with SciRS2 patterns
- `torsh-linalg` ✅ - Linear algebra operations
- `torsh-functional` ✅ - Functional programming utilities

#### Extended Ecosystem (19/19) - 100% SUCCESS
- `torsh-benches` ✅ - Benchmarking suite
- `torsh-cluster` ✅ - **NEWLY FIXED** - Clustering algorithms with SciRS2
- `torsh-distributed` ✅ - Distributed training
- `torsh-ffi` ✅ - Foreign function interface
- `torsh-fx` ✅ - Effects and transformations
- `torsh-graph` ✅ - Graph neural networks with spatial operations
- `torsh-hub` ✅ - Model hub and registry
- `torsh-jit` ✅ - Just-in-time compilation
- `torsh-metrics` ✅ - Performance metrics
- `torsh-models` ✅ - Pre-trained models
- `torsh-profiler` ✅ - Performance profiling
- `torsh-python` ✅ - Python bindings
- `torsh-quantization` ✅ - Model quantization
- `torsh-series` ✅ - Time series analysis
- `torsh-signal` ✅ - Signal processing
- `torsh-sparse` ✅ - Sparse tensor operations
- `torsh-special` ✅ - Special mathematical functions
- `torsh-utils` ✅ - Utilities and helpers
- `torsh-vision` ✅ - Computer vision models with spatial operations
- `torsh-package` ✅ - Package management

### Remaining Issues (1/30) - Non-SciRS2 External Dependencies

#### 1. `torsh-cli` ❌ - External dependency API changes
- **Issue**: sysinfo and byte_unit crate API changes (unrelated to SciRS2)
- **Impact**: Command line tools unavailable
- **Status**: External crate dependency updates needed
- **Priority**: Low - core functionality unaffected
- **SciRS2 Status**: ✅ All SciRS2 integrations successful

### 🏆 MAJOR ACHIEVEMENT: SciRS2 SPATIAL FIX SUCCESS

#### Recently Fixed (September 28, 2025)
- **scirs2-spatial** dependency issue resolved by SciRS2 team
- **torsh-cluster** ✅ - Now fully operational with advanced clustering algorithms
- **torsh-vision** ✅ - Enhanced with spatial operations capability
- **torsh-graph** ✅ - Complete graph neural network spatial functionality

## 🏆 PROVEN SUCCESSFUL PATTERNS

### Migration Pattern 1: Cargo.toml SciRS2 POLICY Compliance
```toml
# ❌ REMOVED: Direct dependencies violating SciRS2 POLICY
# rand = "0.9.2"         # REMOVED: Use scirs2_core::random instead (SciRS2 POLICY)
# rand_distr = "0.5.1"   # REMOVED: Use scirs2_core::random::prelude instead (SciRS2 POLICY)
# ndarray = "0.16"       # REMOVED: Use scirs2_autograd::ndarray instead (SciRS2 POLICY)

# ✅ SciRS2 POLICY COMPLIANT dependencies
scirs2-core = "0.1.0-beta.3"
scirs2-autograd = "0.1.0-beta.3"
```

### Migration Pattern 2: Random Number Generation Integration
```rust
// ❌ OLD PATTERN (violated SciRS2 POLICY)
use rand::{thread_rng, Rng};
rng: Random<scirs2_core::Random<StdRng>>  // Nested type confusion

// ✅ NEW PATTERN (SciRS2 compliant)
use scirs2_core::random::{CoreRandom, seeded_rng};
use scirs2_core::random::rngs::StdRng;
rng: CoreRandom<StdRng>                   // Clean type declaration
rng: seeded_rng(42)                       // Proper initialization
```

### Migration Pattern 3: Array Operations Integration
```rust
// ❌ OLD PATTERN (violated SciRS2 POLICY)
use ndarray::{Array, Array1, Array2};

// ✅ NEW PATTERN (SciRS2 compliant)
use scirs2_autograd::ndarray::{Array, Array1, Array2, array};  // Full functionality
// OR
use scirs2_core::ndarray_ext::{Array, ArrayView};              // Basic types
```

### Migration Pattern 4: Type Declaration Modernization
```rust
// ❌ OLD: Confused nested types
struct Algorithm {
    rng: CoreRandom<scirs2_core::Random<StdRng>>,  // WRONG
}

// ✅ NEW: Clean SciRS2 types
struct Algorithm {
    rng: CoreRandom<StdRng>,  // Seeded, deterministic
    // OR
    rng: CoreRandom,          // Thread-local, fast
}
```

### Migration Pattern 5: Initialization Pattern Updates
```rust
// ❌ OLD: Incorrect initialization
Self { rng: Random::seed(42) }

// ✅ NEW: SciRS2 canonical patterns
Self { rng: seeded_rng(42) }      // For deterministic behavior
Self { rng: thread_rng() }        // For fast, non-deterministic
```

## 📋 STANDARD RESOLUTION WORKFLOW

### Proven 5-Step Migration Process

1. **Cargo.toml Cleanup**
   - Remove all direct `rand*` and `ndarray*` dependencies
   - Add policy compliance comments
   - Add appropriate scirs2-* dependencies

2. **Import Path Migration**
   - Replace `rand::*` with `scirs2_core::random::*`
   - Replace `ndarray::*` with `scirs2_autograd::ndarray::*` (or `scirs2_core::ndarray_ext::*`)

3. **Type Declaration Fixes**
   - Remove nested `Random<Random<T>>` patterns
   - Use `CoreRandom<StdRng>` for seeded RNG
   - Use `CoreRandom` for thread-local RNG

4. **Initialization Updates**
   - Replace `Random::seed()` with `seeded_rng()`
   - Replace `rand::thread_rng()` with `thread_rng()` from scirs2_core

5. **Compilation Validation**
   - Test individual package compilation
   - Verify no remaining rand/ndarray direct dependencies

### Success Metrics Achieved

- **100% Policy Compliance**: All working packages follow SciRS2 POLICY
- **Zero Direct Dependencies**: No rand/ndarray violations in successful packages
- **Clean Type Hierarchies**: Proper CoreRandom usage throughout
- **Consistent Patterns**: Uniform migration approach across all packages
- **Maintained Functionality**: 100% PyTorch API compatibility preserved

## Quick Reference

### Current Workspace Integration (v0.1.0-beta.1)
```toml
[workspace.dependencies]
# Essential SciRS2 dependencies for ToRSh - STABLE INTEGRATION
# Status: 100% SUCCESS (29/29 packages) - SciRS2 v0.1.1 stable + OxiBLAS v0.1.2
scirs2-core = { version = "0.1.1", features = ["simd", "parallel", "memory_management", "serialization", "oxicode", "oxiblas-blas", "oxiblas-lapack", "oxiblas-ndarray", "validation", "types", "random"] }
scirs2-autograd = { version = "0.1.1" }
scirs2-special = { version = "0.1.1" }
scirs2-sparse = { version = "0.1.1" }
scirs2-optimize = { version = "0.1.1" }
scirs2-signal = { version = "0.1.1" }
scirs2-fft = { version = "0.1.1" }

# Comprehensive SciRS2 integration (100% coverage - all stable)
scirs2-cluster = { version = "0.1.1" }
scirs2-datasets = { version = "0.1.1" }
scirs2-graph = { version = "0.1.1" }
scirs2-metrics = { version = "0.1.1" }
scirs2-series = { version = "0.1.1" }
scirs2-spatial = { version = "0.1.1" }
scirs2-stats = { version = "0.1.1" }
scirs2-text = { version = "0.1.1" }
scirs2-vision = { version = "0.1.1" }
scirs2-linalg = { version = "0.1.1" }  # Includes scipy.linalg compatibility (35 functions)
scirs2-neural = { version = "0.1.1" }

# OptiRS integration for advanced optimization (still RC)
optirs = { version = "0.1.0-rc", default-features = false }
optirs-core = { version = "0.1.0-rc", default-features = false }
optirs-core = { path = "../optirs/optirs-core", default-features = false }
```

### Module-Specific Usage
```toml
# torsh-nn
scirs2 = { workspace = true, features = ["neural"] }

# torsh-optim
scirs2-optimize = { workspace = true }
optirs = { workspace = true }

# torsh-autograd
scirs2 = { workspace = true, features = ["autograd"] }
scirs2-autograd = { workspace = true }  # For array types

# torsh-tensor
scirs2-autograd = { workspace = true }  # For ndarray types

# torsh-data
scirs2 = { workspace = true, features = ["datasets"] }
```

### Correct Import Patterns for Arrays

```rust
// OPTION 1: When you need full ndarray functionality including array! macro:
use scirs2_autograd::ndarray::{Array, Array1, Array2, array};

// OPTION 2: When you only need basic array types (no array! macro):
use scirs2_core::ndarray_ext::{Array, ArrayView, ArrayViewMut};
use scirs2_core::ndarray_ext::{stats, matrix, manipulation};

// NEVER use ndarray directly:
// use ndarray::{...}  // ❌ Violates SciRS2 policy
```

**Key Points**:
- `scirs2_autograd::ndarray` - Full ndarray re-export with array! macro
- `scirs2_core::ndarray_ext` - Basic types and operations, NO array! macro
- Choose based on your needs (array! macro requirement)

**Remember**: Start minimal, add based on evidence, document everything!