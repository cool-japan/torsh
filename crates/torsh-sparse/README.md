# torsh-sparse

Sparse tensor operations for ToRSh, leveraging scirs2-sparse for efficient sparse matrix computations.

## Overview

This crate provides comprehensive sparse tensor support:

- **Sparse Formats**: COO, CSR, CSC, and hybrid formats
- **Operations**: Sparse matrix multiplication, addition, transpose
- **Conversions**: Dense to sparse, format conversions
- **GPU Support**: CUDA sparse operations via cuSPARSE
- **Integration**: Seamless integration with scirs2-sparse

## Usage

### Creating Sparse Tensors

```rust
use torsh_sparse::prelude::*;

// From COO format (coordinate list)
let indices = tensor![[0, 1, 1], [2, 0, 2]]; // [[row], [col]]
let values = tensor![3.0, 4.0, 5.0];
let size = vec![3, 4];
let sparse_coo = sparse_coo_tensor(indices, values, size)?;

// From dense tensor
let dense = tensor![[1.0, 0.0, 2.0],
                   [0.0, 0.0, 3.0],
                   [4.0, 5.0, 0.0]];
let sparse = dense.to_sparse()?;

// From CSR format (compressed sparse row)
let crow_indices = tensor![0, 2, 3, 5]; // row pointers
let col_indices = tensor![0, 2, 2, 0, 1]; // column indices
let values = tensor![1.0, 2.0, 3.0, 4.0, 5.0];
let sparse_csr = sparse_csr_tensor(crow_indices, col_indices, values, size)?;
```

### Sparse Operations

```rust
// Sparse matrix multiplication (leveraging scirs2-sparse)
let a = sparse_coo_tensor(indices_a, values_a, size_a)?;
let b = sparse_coo_tensor(indices_b, values_b, size_b)?;
let c = sparse::mm(&a, &b)?;

// Sparse-dense multiplication
let sparse_matrix = load_sparse_matrix()?;
let dense_vector = randn(&[1000]);
let result = sparse::mv(&sparse_matrix, &dense_vector)?;

// Element-wise operations
let sum = sparse::add(&sparse_a, &sparse_b)?;
let product = sparse::mul(&sparse_a, &sparse_b)?;

// Transpose
let transposed = sparse_matrix.t()?;
```

### Format Conversions

```rust
// Convert between formats
let coo = create_coo_tensor()?;
let csr = coo.to_csr()?;
let csc = coo.to_csc()?;

// Convert to dense
let dense = sparse_tensor.to_dense()?;

// Hybrid format for better performance
let hybrid = sparse_tensor.to_hybrid(blocksize=16)?;
```

### Advanced Sparse Operations

```rust
// Sparse linear algebra (via scirs2-sparse)
use torsh_sparse::linalg::*;

// Sparse LU decomposition
let (l, u) = sparse_lu(&sparse_matrix)?;

// Sparse Cholesky decomposition
let l = sparse_cholesky(&symmetric_sparse)?;

// Solve sparse linear system
let x = sparse_solve(&sparse_a, &b)?;

// Iterative solvers
let x = conjugate_gradient(&sparse_a, &b, max_iter=1000, tol=1e-6)?;
let x = gmres(&sparse_a, &b, restart=50, max_iter=1000)?;
```

### Sparse Neural Network Layers

```rust
use torsh_sparse::nn::*;

// Sparse Linear layer
let sparse_linear = SparseLinear::new(
    in_features=1000,
    out_features=100,
    sparsity=0.9, // 90% sparse
);

// Sparse Embedding
let sparse_embedding = SparseEmbedding::new(
    num_embeddings=10000,
    embedding_dim=300,
    sparsity_pattern=block_sparse(block_size=16),
);

// Graph Convolution (for GNNs)
let gcn = GraphConvolution::new(
    in_features=64,
    out_features=32,
    bias=true,
);
```

### GPU Acceleration

```rust
// Move sparse tensor to GPU
let gpu_sparse = sparse_tensor.cuda()?;

// cuSPARSE operations
let result = sparse::cuda::spmm(&gpu_sparse_a, &gpu_dense_b)?;
let result = sparse::cuda::spmv(&gpu_sparse_a, &gpu_vector)?;

// Batched sparse operations
let batch_sparse = create_batch_sparse_tensors()?;
let results = sparse::cuda::batch_spmm(&batch_sparse, &batch_dense)?;
```

### Sparse Patterns and Masks

```rust
// Create structured sparsity patterns
let block_sparse_pattern = SparsityPattern::block_sparse(
    shape=[1024, 1024],
    block_size=16,
);

let banded_pattern = SparsityPattern::banded(
    shape=[1000, 1000],
    bandwidth=5,
);

// Apply sparsity to dense tensor
let sparse = dense_tensor.apply_sparsity(&block_sparse_pattern)?;

// Pruning utilities
let pruned = prune_magnitude(
    &dense_tensor,
    sparsity=0.9,
    structured=true,
)?;
```

### Sparse Gradients

```rust
// Sparse optimizer for sparse gradients
use torsh_sparse::optim::*;

let sparse_adam = SparseAdam::new(
    params,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
);

// Gradient accumulation for sparse tensors
let grad_accumulator = SparseGradientAccumulator::new();
grad_accumulator.accumulate(&sparse_gradients)?;
```

### Utilities

```rust
// Analyze sparsity
let stats = analyze_sparsity(&tensor)?;
println!("Sparsity: {:.2}%", stats.sparsity * 100.0);
println!("NNZ: {}", stats.nnz);
println!("Pattern: {:?}", stats.pattern_type);

// Visualize sparse matrix
sparse::visualize(&sparse_matrix, "sparse_pattern.png")?;

// Benchmark sparse operations
let benchmark = SparseBenchmark::new();
let results = benchmark.compare_formats(&sparse_tensor, &operations)?;
```

## Integration with SciRS2

This crate fully leverages scirs2-sparse for:
- Optimized sparse BLAS operations
- Efficient sparse matrix formats
- Hardware-accelerated sparse computations
- Advanced sparse linear algebra

## Performance Tips

1. Choose the right format for your access pattern
2. Use CSR for row-wise operations, CSC for column-wise
3. Consider hybrid formats for mixed access patterns
4. Use batched operations when possible
5. Profile different sparse formats for your use case

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.