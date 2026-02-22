# Performance Guide for ToRSh Linear Algebra

## Overview

This document provides detailed performance characteristics, optimization strategies, and benchmarking guidelines for torsh-linalg operations. Understanding these aspects is crucial for building efficient numerical applications.

## Computational Complexity

### Matrix Operations

| Operation | Time Complexity | Memory | Notes |
|-----------|----------------|---------|-------|
| Matrix Multiplication (n×n) | O(n³) | O(n²) | Can be optimized to O(n^2.376) |
| Matrix-Vector Product | O(n²) | O(n) | Highly optimized in practice |
| Vector-Matrix Product | O(n²) | O(n) | Similar to matvec |
| Batch Matrix Multiplication | O(b·n³) | O(b·n²) | b = batch size |
| Transpose | O(n²) | O(1) | In-place possible |
| Trace | O(n) | O(1) | Only diagonal access |
| Determinant (small) | O(1) | O(1) | Direct formulas for n ≤ 3 |
| Determinant (large) | O(n³) | O(n²) | Via LU decomposition |

### Matrix Decompositions

| Decomposition | Time Complexity | Memory | Stability |
|---------------|----------------|---------|-----------|
| LU (with pivoting) | O(2n³/3) | O(n²) | Backward stable |
| QR (Gram-Schmidt) | O(2n³/3) | O(n²) | Stable with modification |
| Cholesky | O(n³/3) | O(n²) | Inherently stable |
| SVD (full) | O(4n³) | O(n²) | High accuracy |
| SVD (power iteration) | O(k·n²·iter) | O(n²) | k = num singular values |
| Eigendecomposition | O(k·n²·iter) | O(n²) | k = num eigenvalues |
| Polar Decomposition | O(4n³) | O(n²) | Via SVD |
| Schur Decomposition | O(10n³) | O(n²) | QR iteration |
| Jordan Form | O(k·n²·iter) | O(n²) | Limited to simple cases |

### Linear System Solving

| Method | Time Complexity | Memory | Best For |
|--------|----------------|---------|----------|
| LU Solve | O(n³ + n²) | O(n²) | General dense systems |
| Cholesky Solve | O(n³/3 + n²) | O(n²) | SPD systems |
| QR Solve | O(2n³/3 + n²) | O(n²) | Overdetermined systems |
| Triangular Solve | O(n²) | O(1) | Pre-factored systems |
| Thomas Algorithm | O(n) | O(1) | Tridiagonal systems |
| Pentadiagonal Solver | O(n) | O(1) | 5-diagonal systems |
| Conjugate Gradient | O(n² × iter) | O(n) | Large sparse SPD |
| Multigrid V-cycle | O(n) | O(n) | Elliptic PDEs |
| Multigrid W-cycle | O(n log n) | O(n) | More robust |

### Matrix Functions

| Function | Time Complexity | Memory | Method |
|----------|----------------|---------|---------|
| Matrix Inverse | O(n³) | O(n²) | Via LU decomposition |
| Pseudoinverse | O(4n³) | O(n²) | Via SVD |
| Matrix Exponential | O(n³) | O(n²) | Padé approximation |
| Matrix Logarithm | O(n³) | O(n²) | Eigendecomposition |
| Matrix Square Root | O(n³) | O(n²) | Eigendecomposition |
| Matrix Power | O(n³) | O(n²) | Eigendecomposition |
| Matrix Norm (Frobenius) | O(n²) | O(1) | Sum of squares |
| Matrix Norm (2-norm) | O(4n³) | O(n²) | Via SVD |

## Memory Usage Patterns

### Storage Requirements

#### Dense Matrices
```rust
// Memory usage for n×n float32 matrix
let matrix_size_bytes = n * n * 4; // 4 bytes per float32
let additional_workspace = matrix_size_bytes; // Typical for decompositions
```

#### Workspace Requirements
- **LU Decomposition**: 1 additional matrix + pivot vector
- **QR Decomposition**: 1 additional matrix + reflection vectors  
- **SVD**: 3 matrices (U, Σ, V^T) for full decomposition
- **Eigendecomposition**: 2 matrices (eigenvalues + eigenvectors)

#### Memory Access Patterns
- **Row-major**: Better for C/C++ interop, cache-friendly for row operations
- **Column-major**: Better for Fortran interop, cache-friendly for column operations
- **Block access**: Optimal for large matrices, exploits cache hierarchy

### Cache Optimization Strategies

#### Block Algorithms
```rust
// Conceptual blocked matrix multiplication
fn blocked_matmul(a: &Matrix, b: &Matrix, c: &mut Matrix, block_size: usize) {
    for i in (0..n).step_by(block_size) {
        for j in (0..n).step_by(block_size) {
            for k in (0..n).step_by(block_size) {
                // Multiply blocks that fit in cache
                multiply_block(a, b, c, i, j, k, block_size);
            }
        }
    }
}
```

#### Cache-Friendly Patterns
1. **Temporal Locality**: Reuse data while in cache
2. **Spatial Locality**: Access contiguous memory
3. **Loop Tiling**: Process data in cache-sized blocks

## Performance Optimization Strategies

### Algorithm Selection

#### Matrix Size Considerations
```rust
fn optimal_algorithm_selection(n: usize) -> Algorithm {
    match n {
        0..=3 => Algorithm::DirectFormula,     // Hardcoded formulas
        4..=32 => Algorithm::SimpleIterative,  // Basic implementations
        33..=1000 => Algorithm::Blocked,       // Cache-friendly blocks
        1001..=10000 => Algorithm::Parallel,   // Multi-threaded
        _ => Algorithm::OutOfCore,             // Disk-based if needed
    }
}
```

#### Problem Structure Exploitation
- **Symmetric**: Use only upper/lower triangle (2× memory savings)
- **Positive Definite**: Use Cholesky instead of LU (2× faster)
- **Tridiagonal**: Use specialized O(n) algorithms
- **Sparse**: Use iterative methods with preconditioning

### Memory Optimization

#### In-Place Operations
```rust
// Prefer in-place operations when possible
matrix.cholesky_inplace()?;              // Overwrites input
let l = matrix.cholesky(false)?;         // Creates new matrix
```

#### Memory Pooling
```rust
// Reuse workspace arrays across operations
struct WorkspacePool {
    matrices: Vec<Matrix>,
    vectors: Vec<Vector>,
}
```

#### Data Layout Optimization
- **Structure of Arrays (SoA)**: Better for vectorization
- **Array of Structures (AoS)**: Better for object-oriented access
- **Hybrid Layouts**: Block-based approaches for large matrices

### Parallelization Strategies

#### Thread-Level Parallelism
```rust
// Conceptual parallel matrix multiplication
use rayon::prelude::*;

fn parallel_matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut c = Matrix::zeros(a.rows(), b.cols());
    
    c.chunks_mut(BLOCK_SIZE)
        .enumerate()
        .par_bridge()
        .for_each(|(i, block)| {
            compute_block(a, b, block, i);
        });
    
    c
}
```

#### SIMD Optimization
- **Vector Instructions**: Process multiple elements simultaneously
- **Auto-vectorization**: Compiler optimizations
- **Manual SIMD**: Explicit vector intrinsics for critical loops

### Numerical Optimization

#### Reduced Precision
```rust
// Mixed precision: float32 for speed, float64 for accuracy
fn mixed_precision_solve(a: &Matrix<f64>, b: &Vector<f64>) -> Vector<f64> {
    // Fast solve in float32
    let a32 = a.to_f32();
    let b32 = b.to_f32();
    let x32 = solve(&a32, &b32)?;
    
    // Refine in float64
    iterative_refinement(a, b, &x32.to_f64(), 2)
}
```

#### Early Termination
```rust
// Stop iterations when converged
fn adaptive_iteration(tolerance: f32, max_iter: usize) {
    for i in 0..max_iter {
        let residual = compute_residual();
        if residual < tolerance {
            return (solution, i); // Early termination
        }
    }
}
```

## Benchmarking Guidelines

### Performance Metrics

#### Timing Measurements
```rust
use std::time::Instant;

fn benchmark_operation<F>(op: F, name: &str) -> Duration 
where F: Fn() -> Result<()> {
    let start = Instant::now();
    op()?;
    let duration = start.elapsed();
    println!("{}: {:?}", name, duration);
    duration
}
```

#### FLOPS Calculation
```rust
fn theoretical_flops(operation: Operation, n: usize) -> u64 {
    match operation {
        Operation::MatMul => 2 * n * n * n,           // 2n³
        Operation::LU => (2 * n * n * n) / 3,         // 2n³/3
        Operation::Cholesky => (n * n * n) / 3,       // n³/3
        Operation::QR => (4 * n * n * n) / 3,         // 4n³/3
        // ... other operations
    }
}
```

#### Memory Bandwidth
```rust
fn memory_bandwidth(bytes_transferred: u64, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    (bytes_transferred as f64) / seconds / 1e9 // GB/s
}
```

### Benchmark Suites

#### Matrix Sizes for Testing
```rust
const BENCHMARK_SIZES: &[usize] = &[
    10, 32, 100, 316,     // Small matrices
    1000, 3162, 10000,    // Medium matrices  
    31622, 100000,        // Large matrices (if memory allows)
];
```

#### Problem Types
1. **Well-conditioned**: Random matrices with κ(A) ≈ 10
2. **Ill-conditioned**: Hilbert matrices with κ(A) ≈ 10^k
3. **Structured**: Tridiagonal, Toeplitz, circulant matrices
4. **Sparse**: Random sparse with controlled fill-in

### Performance Baselines

#### Reference Implementations
Compare against:
- **BLAS/LAPACK**: Industry standard
- **Intel MKL**: Optimized commercial library
- **OpenBLAS**: Open-source optimized BLAS
- **Eigen**: C++ template library

#### Expected Performance
```rust
// Typical performance ranges for modern hardware
struct PerformanceTargets {
    matmul_gflops: f32,      // 100-1000+ GFLOPS depending on size
    memory_bandwidth: f32,    // 10-100 GB/s depending on access pattern
    solver_efficiency: f32,   // 80-95% of peak for large problems
}
```

## Platform-Specific Optimizations

### CPU Architectures

#### x86_64 Optimizations
- **AVX/AVX2**: 256-bit SIMD instructions
- **FMA**: Fused multiply-add operations
- **Cache Prefetching**: Explicit prefetch instructions

#### ARM Optimizations
- **NEON**: ARM SIMD instruction set
- **SVE**: Scalable Vector Extensions (newer ARM)
- **Memory Ordering**: Relaxed consistency models

### Compiler Optimizations

#### Optimization Flags
```toml
# Cargo.toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
```

#### Target-Specific Features
```bash
# Build with native CPU features
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Memory Hierarchy Considerations

### Cache Levels
- **L1 Cache**: 32-64 KB, 1-2 cycles latency
- **L2 Cache**: 256 KB - 1 MB, 10-20 cycles latency  
- **L3 Cache**: 8-32 MB, 40-100 cycles latency
- **Main Memory**: GB scale, 200-300 cycles latency

### Optimization Strategies
1. **Blocking**: Fit working set in cache level
2. **Prefetching**: Load data before needed
3. **Loop Fusion**: Reduce memory traffic
4. **Data Reuse**: Maximize temporal locality

## Scalability Considerations

### Problem Size Scaling
```rust
// Expected scaling behavior
fn expected_runtime(n: usize, base_time: f64, algorithm: Algorithm) -> f64 {
    match algorithm {
        Algorithm::Linear => base_time * (n as f64),                    // O(n)
        Algorithm::Quadratic => base_time * (n as f64).powi(2),         // O(n²)
        Algorithm::Cubic => base_time * (n as f64).powi(3),             // O(n³)
        Algorithm::Linearithmic => base_time * (n as f64) * (n as f64).ln(), // O(n log n)
    }
}
```

### Parallel Scaling
- **Strong Scaling**: Fixed problem size, increasing processors
- **Weak Scaling**: Problem size increases with processors
- **Amdahl's Law**: Theoretical speedup limits due to serial portions

## Best Practices Summary

### Algorithm Selection
1. Match algorithm to problem structure
2. Consider problem size and available memory
3. Use specialized algorithms for structured matrices
4. Prefer numerically stable algorithms

### Implementation Optimization
1. Minimize memory allocations in hot paths
2. Use in-place operations when possible
3. Exploit cache hierarchy with blocking
4. Consider SIMD and parallel opportunities

### Performance Monitoring
1. Profile before optimizing
2. Measure FLOPS and memory bandwidth
3. Compare against theoretical peaks
4. Test across different problem sizes and types

### Memory Management
1. Reuse workspace arrays
2. Consider memory layout for access patterns
3. Monitor peak memory usage
4. Use appropriate data types (f32 vs f64)