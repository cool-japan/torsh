# ToRSh Sparse Tensor Guide

## Overview

ToRSh-Sparse is a comprehensive sparse tensor library for Rust that provides PyTorch-compatible APIs with superior performance. This guide covers the fundamentals of sparse tensors, their formats, operations, and best practices.

## Table of Contents

1. [Introduction to Sparse Tensors](#introduction-to-sparse-tensors)
2. [Sparse Tensor Formats](#sparse-tensor-formats)
3. [Creating Sparse Tensors](#creating-sparse-tensors)
4. [Basic Operations](#basic-operations)
5. [Advanced Operations](#advanced-operations)
6. [Neural Network Integration](#neural-network-integration)
7. [Performance Optimization](#performance-optimization)
8. [Memory Management](#memory-management)
9. [Interoperability](#interoperability)
10. [Best Practices](#best-practices)

## Introduction to Sparse Tensors

Sparse tensors are data structures that efficiently store and process tensors with a large number of zero elements. They are essential for:

- **Memory Efficiency**: Only non-zero elements are stored
- **Computational Efficiency**: Operations skip zero elements
- **Numerical Stability**: Preserved sparsity patterns
- **Domain Applications**: Graph neural networks, NLP, computer vision

### When to Use Sparse Tensors

- **High Sparsity**: When >90% of elements are zero
- **Large Dimensions**: Matrices with millions of elements
- **Structured Patterns**: Diagonal, banded, or block structures
- **Graph Data**: Adjacency matrices, attention patterns

## Sparse Tensor Formats

ToRSh-Sparse supports multiple sparse formats, each optimized for different use cases:

### COO (Coordinate Format)
- **Best for**: Construction, format conversion
- **Structure**: Three arrays (row, col, values)
- **Advantages**: Easy to construct, efficient for random access
- **Disadvantages**: Not optimal for arithmetic operations

```rust
use torsh_sparse::{SparseTensor, COOTensor};

let rows = vec![0, 1, 2];
let cols = vec![0, 1, 2]; 
let values = vec![1.0, 2.0, 3.0];
let coo = COOTensor::new(rows, cols, values, (3, 3))?;
```

### CSR (Compressed Sparse Row)
- **Best for**: Matrix-vector multiplication, row-wise operations
- **Structure**: Row pointers, column indices, values
- **Advantages**: Efficient matvec, cache-friendly row access
- **Disadvantages**: Expensive column access

```rust
use torsh_sparse::CSRTensor;

let row_ptrs = vec![0, 1, 2, 3];
let col_indices = vec![0, 1, 2];
let values = vec![1.0, 2.0, 3.0];
let csr = CSRTensor::new(row_ptrs, col_indices, values, (3, 3))?;
```

### CSC (Compressed Sparse Column)
- **Best for**: Matrix-vector multiplication (transpose), column-wise operations
- **Structure**: Column pointers, row indices, values
- **Advantages**: Efficient column access, transpose operations
- **Disadvantages**: Expensive row access

```rust
use torsh_sparse::CSCTensor;

let col_ptrs = vec![0, 1, 2, 3];
let row_indices = vec![0, 1, 2];
let values = vec![1.0, 2.0, 3.0];
let csc = CSCTensor::new(col_ptrs, row_indices, values, (3, 3))?;
```

### BSR (Block Sparse Row)
- **Best for**: Block-structured matrices, dense subblocks
- **Structure**: Block pointers, block indices, dense blocks
- **Advantages**: Vectorized operations on blocks
- **Disadvantages**: Memory overhead for small blocks

```rust
use torsh_sparse::BSRTensor;

let block_ptrs = vec![0, 1, 2];
let block_indices = vec![0, 1];
let blocks = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 blocks
let bsr = BSRTensor::new(block_ptrs, block_indices, blocks, (2, 2), (4, 4))?;
```

### DIA (Diagonal Format)
- **Best for**: Diagonal and banded matrices
- **Structure**: Diagonal data, offset array
- **Advantages**: Excellent for banded operations
- **Disadvantages**: Wastes memory for irregular patterns

```rust
use torsh_sparse::DIATensor;

let diagonals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let offsets = vec![-1, 0, 1]; // sub-diagonal, main diagonal, super-diagonal
let dia = DIATensor::new(diagonals, offsets, (3, 3))?;
```

### ELL (ELLPACK)
- **Best for**: Regular sparsity patterns, GPU computation
- **Structure**: Padded column indices, padded values
- **Advantages**: Coalesced memory access, vectorization
- **Disadvantages**: Memory waste for irregular patterns

```rust
use torsh_sparse::ELLTensor;

let indices = vec![0, 1, 2, 0, 1, 2]; // padded column indices
let values = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]; // padded values
let ell = ELLTensor::new(indices, values, 2, (3, 3))?; // max 2 non-zeros per row
```

### DSR (Dynamic Sparse Row)
- **Best for**: Dynamic sparsity patterns, frequent updates
- **Structure**: BTreeMap-based storage per row
- **Advantages**: Efficient insertion/deletion, sorted access
- **Disadvantages**: Memory overhead, slower bulk operations

```rust
use torsh_sparse::DSRTensor;

let mut dsr = DSRTensor::new((3, 3))?;
dsr.set(0, 0, 1.0)?;
dsr.set(1, 1, 2.0)?;
dsr.set(2, 2, 3.0)?;
```

## Creating Sparse Tensors

### From Dense Tensors

```rust
use torsh_sparse::{SparseTensor, COOTensor};
use torsh_tensor::Tensor;

// Create dense tensor
let dense = Tensor::zeros(&[3, 3], DType::F32, Device::CPU)?;
dense.set(&[0, 0], 1.0)?;
dense.set(&[1, 1], 2.0)?;
dense.set(&[2, 2], 3.0)?;

// Convert to sparse
let sparse = COOTensor::from_dense(&dense)?;
```

### From Triplets

```rust
use torsh_sparse::COOTensor;

// Create from coordinate triplets
let triplets = vec![
    (0, 0, 1.0),
    (1, 1, 2.0),
    (2, 2, 3.0),
];

let sparse = COOTensor::from_triplets(triplets, (3, 3))?;
```

### From Builder Pattern

```rust
use torsh_sparse::{SparseTensorBuilder, SparseFormat};

let sparse = SparseTensorBuilder::new((1000, 1000))
    .format(SparseFormat::CSR)
    .device(Device::CPU)
    .dtype(DType::F32)
    .reserve(10000) // Reserve space for 10k non-zeros
    .build()?;
```

## Basic Operations

### Element Access

```rust
// Get element
let value = sparse.get(0, 0)?;

// Set element (for mutable formats)
sparse.set(0, 0, 5.0)?;

// Check if element exists
let exists = sparse.contains(0, 0);
```

### Format Conversion

```rust
use torsh_sparse::{COOTensor, CSRTensor, CSCTensor};

// Convert between formats
let coo = COOTensor::from_triplets(triplets, (3, 3))?;
let csr = CSRTensor::from_coo(&coo)?;
let csc = CSCTensor::from_csr(&csr)?;

// Or use unified interface
let unified = UnifiedSparseTensor::from_coo(coo)?;
let optimized = unified.optimize_for_operation(OperationType::MatVec)?;
```

### Arithmetic Operations

```rust
// Addition
let result = sparse_a.add(&sparse_b)?;

// Scalar multiplication
let scaled = sparse.scale(2.0)?;

// Matrix multiplication
let product = sparse_a.matmul(&sparse_b)?;

// Element-wise operations
let element_wise = sparse_a.mul_element_wise(&sparse_b)?;
```

### Reductions

```rust
// Sum all elements
let total = sparse.sum()?;

// Sum along axis
let row_sums = sparse.sum_axis(1)?;

// Norm calculations
let l2_norm = sparse.norm(2.0)?;
let frobenius = sparse.frobenius_norm()?;

// Diagonal extraction
let diagonal = sparse.diagonal()?;
```

## Advanced Operations

### Linear Algebra

```rust
use torsh_sparse::linalg::*;

// Solve linear system Ax = b
let x = conjugate_gradient(&A, &b, 1e-6, 1000)?;

// Solve with preconditioner
let preconditioner = incomplete_lu(&A, 0.01)?;
let x = conjugate_gradient_preconditioned(&A, &b, &preconditioner, 1e-6, 1000)?;

// Find largest eigenvalue
let (eigenvalue, eigenvector) = power_iteration(&A, 1e-6, 1000)?;

// Factorization
let lu = incomplete_lu(&A, 0.01)?;
let (L, U) = lu.factors();
```

### Pattern Analysis

```rust
use torsh_sparse::pattern_analysis::*;

// Analyze sparsity patterns
let analysis = analyze_sparsity_pattern(&sparse)?;
println!("Density: {:.2}%", analysis.density * 100.0);
println!("Bandwidth: {}", analysis.bandwidth);

// Matrix reordering
let (reordered, permutation) = rcm_reordering(&sparse)?;

// Detect special patterns
let patterns = detect_patterns(&sparse)?;
if patterns.is_diagonal {
    println!("Matrix is diagonal");
}
if patterns.is_banded {
    println!("Matrix is banded with bandwidth {}", patterns.bandwidth);
}
```

### Memory Management

```rust
use torsh_sparse::memory_management::*;

// Create memory-aware sparse tensor
let sparse = MemoryAwareBuilder::new((10000, 10000))
    .memory_budget(1_000_000_000) // 1GB budget
    .build()?;

// Monitor memory usage
let stats = sparse.memory_stats();
println!("Memory usage: {} bytes", stats.total_bytes);
println!("Compression ratio: {:.2}x", stats.compression_ratio);

// Garbage collection
sparse.gc()?;
```

## Neural Network Integration

### Sparse Layers

```rust
use torsh_sparse::nn::*;

// Sparse linear layer
let sparse_linear = SparseLinear::new(784, 10, 0.9)?; // 90% sparsity

// Sparse convolution
let sparse_conv = SparseConv2d::new(3, 64, 3, 0.8)?; // 80% sparsity

// Sparse attention
let sparse_attention = SparseAttention::new(512, 8, 0.95)?; // 95% sparsity
```

### Graph Neural Networks

```rust
use torsh_sparse::nn::GraphConvolution;

// Create graph convolution layer
let gcn = GraphConvolution::new(128, 64, true)?; // with self-loops

// Forward pass with adjacency matrix
let output = gcn.forward(&features, &adjacency_matrix)?;
```

### Pruning

```rust
use torsh_sparse::nn::pruning::*;

// Magnitude-based pruning
let pruned_weights = magnitude_pruning(&weights, 0.9)?; // Keep top 10%

// Structured pruning
let pruned_model = structured_pruning(&model, 0.8)?;
```

## Performance Optimization

### Automatic Format Selection

```rust
use torsh_sparse::{UnifiedSparseTensor, OperationType};

// Automatic optimization
let unified = UnifiedSparseTensor::from_coo(coo)?;
let optimized = unified.optimize_for_operation(OperationType::MatVec)?;

// Manual format selection
let best_format = auto_select_format(&sparse, &[OperationType::MatVec, OperationType::Transpose])?;
```

### Performance Profiling

```rust
use torsh_sparse::performance_tools::*;

// Profile operations
let profiler = PerformanceProfiler::new();
profiler.start_timing("matrix_multiplication");
let result = sparse_a.matmul(&sparse_b)?;
profiler.end_timing("matrix_multiplication");

// Auto-tuning
let tuner = AutoTuner::new();
let optimal_params = tuner.optimize_operation(&sparse, OperationType::MatVec)?;
```

### Memory Optimization

```rust
use torsh_sparse::memory_management::*;

// Memory pool usage
let pool = MemoryPool::new(1_000_000_000)?; // 1GB pool
let sparse = pool.allocate_tensor((10000, 10000))?;

// Memory-aware operations
let result = sparse_a.matmul_memory_efficient(&sparse_b, &pool)?;
```

## Interoperability

### Python/SciPy Integration

```rust
use torsh_sparse::scipy_sparse::*;

// Convert to SciPy format
let scipy_data = to_scipy_sparse(&sparse)?;

// Generate Python code
let python_code = generate_python_code(&sparse, "my_matrix")?;
```

### MATLAB Integration

```rust
use torsh_sparse::matlab_compat::*;

// Export to MATLAB
export_to_matlab(&sparse, "matrix.mat")?;

// Generate MATLAB script
let matlab_script = generate_matlab_script(&sparse, "my_matrix")?;
```

### HDF5 Integration

```rust
use torsh_sparse::hdf5_support::*;

// Save to HDF5
save_sparse_hdf5(&sparse, "data.h5", "matrix")?;

// Load from HDF5
let loaded = load_sparse_hdf5("data.h5", "matrix")?;
```

## Best Practices

### Format Selection Guidelines

1. **COO**: Use for construction and one-time operations
2. **CSR**: Use for row-wise operations and matrix-vector multiplication
3. **CSC**: Use for column-wise operations and transpose multiplication
4. **BSR**: Use for block-structured matrices with dense subblocks
5. **DIA**: Use for diagonal and banded matrices
6. **ELL**: Use for regular patterns and GPU computation
7. **DSR**: Use for dynamic sparsity patterns

### Performance Tips

1. **Choose the right format** for your access patterns
2. **Reuse tensors** to avoid allocation overhead
3. **Use memory pools** for large-scale computations
4. **Profile your code** to identify bottlenecks
5. **Consider hybrid approaches** for complex patterns

### Memory Management

1. **Monitor memory usage** with built-in tools
2. **Use compression** for storage-bound applications
3. **Implement garbage collection** for long-running processes
4. **Consider memory-mapped files** for very large matrices

### Numerical Stability

1. **Use appropriate preconditioners** for iterative methods
2. **Monitor condition numbers** for linear systems
3. **Consider pivoting** for factorizations
4. **Validate convergence** in iterative algorithms

## Error Handling

ToRSh-Sparse uses comprehensive error handling:

```rust
use torsh_sparse::{TorshError, Result};

// Handle specific errors
match sparse.matmul(&other) {
    Ok(result) => println!("Success: {:?}", result),
    Err(TorshError::DimensionMismatch { expected, actual }) => {
        println!("Dimension mismatch: expected {:?}, got {:?}", expected, actual);
    }
    Err(TorshError::UnsupportedOperation { op }) => {
        println!("Unsupported operation: {}", op);
    }
    Err(e) => println!("Other error: {:?}", e),
}
```

## Conclusion

ToRSh-Sparse provides a comprehensive, high-performance sparse tensor library for Rust. With its multiple format support, advanced operations, and seamless integration with the ToRSh ecosystem, it enables efficient sparse computation for machine learning, scientific computing, and graph processing applications.

For more examples and detailed API documentation, see the [API Reference](API_REFERENCE.md) and [Examples](examples/) directory.