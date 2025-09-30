# torsh-linalg

Linear algebra operations for ToRSh, leveraging scirs2-linalg for optimized implementations.

## Overview

This crate provides comprehensive linear algebra functionality by wrapping scirs2-linalg with a PyTorch-compatible API:

- **Matrix Operations**: Multiplication, decompositions, solving
- **Eigenvalue Problems**: Eigenvalues, eigenvectors, SVD
- **Matrix Functions**: Inverse, determinant, trace, norms
- **Specialized Solvers**: Linear systems, least squares, Cholesky
- **Tensor Operations**: Einstein summation, tensor contractions

## Usage

### Basic Matrix Operations

```rust
use torsh_linalg::prelude::*;
use torsh_tensor::prelude::*;

// Matrix multiplication
let a = randn(&[10, 20]);
let b = randn(&[20, 30]);
let c = linalg::matmul(&a, &b)?;

// Batch matrix multiplication
let batch_a = randn(&[32, 10, 20]);
let batch_b = randn(&[32, 20, 30]);
let batch_c = linalg::bmm(&batch_a, &batch_b)?;

// Matrix-vector multiplication
let matrix = randn(&[10, 20]);
let vector = randn(&[20]);
let result = linalg::mv(&matrix, &vector)?;
```

### Decompositions

```rust
// LU decomposition
let (lu, pivots) = linalg::lu(&matrix)?;
let (p, l, u) = linalg::lu_factor(&matrix)?;

// QR decomposition
let (q, r) = linalg::qr(&matrix)?;

// Cholesky decomposition (for positive definite matrices)
let l = linalg::cholesky(&pos_def_matrix)?;

// Eigenvalue decomposition
let (eigenvalues, eigenvectors) = linalg::eig(&square_matrix)?;

// Singular Value Decomposition (SVD)
let (u, s, v) = linalg::svd(&matrix)?;
let (u_reduced, s_reduced, v_reduced) = linalg::svd(&matrix, false)?; // reduced SVD
```

### Solving Linear Systems

```rust
// Solve Ax = b
let a = randn(&[10, 10]);
let b = randn(&[10, 5]);
let x = linalg::solve(&a, &b)?;

// Solve triangular system
let lower = linalg::tril(&a);
let x = linalg::solve_triangular(&lower, &b, true, false)?;

// Least squares solution
let a = randn(&[20, 10]); // overdetermined
let b = randn(&[20]);
let x = linalg::lstsq(&a, &b)?;

// Solve with Cholesky (for positive definite systems)
let x = linalg::cholesky_solve(&pos_def_matrix, &b)?;
```

### Matrix Properties

```rust
// Determinant
let det = linalg::det(&square_matrix)?;

// Inverse
let inv = linalg::inv(&square_matrix)?;
let pinv = linalg::pinv(&matrix)?; // pseudo-inverse

// Matrix norms
let frobenius = linalg::norm(&matrix, "fro")?;
let nuclear = linalg::norm(&matrix, "nuc")?;
let spectral = linalg::norm(&matrix, 2)?;

// Condition number
let cond = linalg::cond(&matrix, None)?;

// Rank
let rank = linalg::matrix_rank(&matrix, None)?;

// Trace
let trace = linalg::trace(&square_matrix)?;
```

### Advanced Operations

```rust
// Einstein summation
let result = linalg::einsum("ij,jk->ik", &[&a, &b])?;
let batch_result = linalg::einsum("bij,bjk->bik", &[&batch_a, &batch_b])?;

// Kronecker product
let kron = linalg::kron(&a, &b)?;

// Matrix exponential
let exp_matrix = linalg::matrix_exp(&square_matrix)?;

// Matrix power
let matrix_squared = linalg::matrix_power(&square_matrix, 2)?;
let matrix_sqrt = linalg::matrix_power(&pos_def_matrix, 0.5)?;
```

### Special Matrix Constructors

```rust
// Identity matrix
let eye = linalg::eye(10, None, None)?;

// Diagonal matrix
let diag_vals = tensor![1.0, 2.0, 3.0, 4.0];
let diag_matrix = linalg::diag(&diag_vals)?;

// Extract diagonal
let diagonal = linalg::diag(&matrix)?;

// Vandermonde matrix
let x = tensor![1.0, 2.0, 3.0, 4.0];
let vander = linalg::vander(&x, None, true)?;
```

### Batch Operations

All operations support batched inputs:

```rust
// Batch inverse
let batch_matrices = randn(&[32, 10, 10]);
let batch_inv = linalg::inv(&batch_matrices)?;

// Batch solve
let batch_a = randn(&[32, 10, 10]);
let batch_b = randn(&[32, 10, 5]);
let batch_x = linalg::solve(&batch_a, &batch_b)?;

// Batch eigenvalues
let batch_eigenvalues = linalg::eigvals(&batch_matrices)?;
```

### Performance Considerations

This crate leverages scirs2-linalg which uses:
- Optimized BLAS/LAPACK implementations
- Multi-threading for large operations
- GPU acceleration when available
- Efficient memory layouts

## Integration with SciRS2

All operations are implemented via scirs2-linalg, ensuring:
- Consistent numerical behavior
- Optimized performance
- Hardware acceleration support
- Compatibility with the scirs2 ecosystem

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.