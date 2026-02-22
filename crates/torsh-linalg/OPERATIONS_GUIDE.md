# ToRSh Linear Algebra Operations Guide

## Overview

The torsh-linalg crate provides a comprehensive suite of linear algebra operations built on top of the ToRSh tensor framework. This guide covers all available operations, their usage patterns, and practical examples.

## Core Operations

### Basic Matrix Operations

#### Matrix Multiplication
```rust
use torsh_linalg::{matmul, matvec, vecmat, bmm};

// Standard matrix multiplication
let result = matmul(&matrix_a, &matrix_b)?;

// Matrix-vector multiplication
let result = matvec(&matrix, &vector)?;

// Vector-matrix multiplication  
let result = vecmat(&vector, &matrix)?;

// Batch matrix multiplication (3D tensors)
let result = bmm(&batch_a, &batch_b)?;
```

#### Matrix Properties
```rust
use torsh_linalg::{det, trace, matrix_rank, cond};

// Determinant calculation
let determinant = det(&matrix)?;

// Matrix trace (sum of diagonal elements)
let trace_value = trace(&matrix)?;

// Matrix rank with optional tolerance
let rank = matrix_rank(&matrix, Some(1e-8))?;

// Condition number (various norms supported)
let condition = cond(&matrix, Some("2"))?; // 2-norm
let condition = cond(&matrix, Some("1"))?; // 1-norm
let condition = cond(&matrix, Some("inf"))?; // infinity-norm
```

### Matrix Decompositions

#### LU Decomposition
```rust
use torsh_linalg::decomposition::lu;

// LU decomposition with partial pivoting
let (p, l, u) = lu(&matrix)?;
// Where: P*A = L*U
// P: permutation matrix
// L: lower triangular matrix
// U: upper triangular matrix
```

#### QR Decomposition
```rust
use torsh_linalg::decomposition::qr;

// QR decomposition using Gram-Schmidt
let (q, r) = qr(&matrix)?;
// Where: A = Q*R
// Q: orthogonal matrix
// R: upper triangular matrix
```

#### Singular Value Decomposition (SVD)
```rust
use torsh_linalg::decomposition::svd;

// Full SVD
let (u, s, vt) = svd(&matrix, true)?;
// Where: A = U*S*V^T

// Compact SVD (economy size)
let (u, s, vt) = svd(&matrix, false)?;
```

#### Eigenvalue Decomposition
```rust
use torsh_linalg::decomposition::{eig, eigvals};

// Full eigendecomposition
let (eigenvalues, eigenvectors) = eig(&matrix)?;

// Eigenvalues only
let eigenvalues = eigvals(&matrix)?;
```

#### Cholesky Decomposition
```rust
use torsh_linalg::decomposition::cholesky;

// Lower triangular Cholesky
let l = cholesky(&matrix, false)?; // A = L*L^T

// Upper triangular Cholesky
let u = cholesky(&matrix, true)?;  // A = U^T*U
```

#### Advanced Decompositions
```rust
use torsh_linalg::decomposition::{polar, schur, jordan_form};

// Polar decomposition
let (p, u) = polar(&matrix, false)?; // A = P*U (right polar)
let (u, p) = polar(&matrix, true)?;  // A = U*P (left polar)

// Schur decomposition
let (q, t) = schur(&matrix)?; // A = Q*T*Q^H

// Jordan canonical form
let (p, j) = jordan_form(&matrix)?; // A = P*J*P^(-1)
```

### Linear System Solving

#### Direct Solvers
```rust
use torsh_linalg::solve::{solve, solve_triangular, inv, pinv};

// General linear system Ax = b
let x = solve(&a, &b)?;

// Triangular system solving
let x = solve_triangular(&l, &b, false, false)?; // Lower triangular
let x = solve_triangular(&u, &b, true, false)?;  // Upper triangular

// Matrix inverse
let a_inv = inv(&a)?;

// Moore-Penrose pseudoinverse
let a_pinv = pinv(&a, Some(1e-8))?;
```

#### Iterative Solvers
```rust
use torsh_linalg::sparse::{cg, DiagonalPreconditioner};

// Conjugate Gradient for symmetric positive definite systems
let x = cg(&a, &b, None, Some(1e-8), Some(1000))?;

// With preconditioning
let precond = DiagonalPreconditioner::new(&a)?;
let x = cg(&a, &b, Some(&precond), Some(1e-8), Some(1000))?;
```

#### Specialized Solvers
```rust
use torsh_linalg::solve::{
    thomas_algorithm, pentadiagonal_solver, 
    toeplitz_solve, vandermonde_solve
};

// Tridiagonal systems (O(n) complexity)
let x = thomas_algorithm(&diag, &lower, &upper, &b)?;

// Pentadiagonal systems
let x = pentadiagonal_solver(&matrix, &b)?;

// Structured matrix solvers
let x = toeplitz_solve(&toeplitz_matrix, &b)?;
let x = vandermonde_solve(&vandermonde_matrix, &b)?;
```

#### Multigrid Solvers
```rust
use torsh_linalg::solve::{multigrid_solve, MultigridConfig, CycleType};

// Configure multigrid solver
let config = MultigridConfig {
    cycle_type: CycleType::VCycle,
    max_levels: 10,
    smoother_iterations: 2,
    tolerance: 1e-8,
    max_iterations: 100,
};

let x = multigrid_solve(&a, &b, &config)?;
```

### Matrix Functions

#### Elementary Functions
```rust
use torsh_linalg::matrix_functions::{
    matrix_exp, matrix_log, matrix_sqrt, matrix_power
};

// Matrix exponential
let exp_a = matrix_exp(&matrix)?;

// Matrix logarithm
let log_a = matrix_log(&matrix)?;

// Matrix square root
let sqrt_a = matrix_sqrt(&matrix)?;

// Matrix power
let a_pow_n = matrix_power(&matrix, 3.5)?;
```

#### Matrix Norms
```rust
use torsh_linalg::matrix_functions::matrix_norm;

// Various matrix norms
let fro_norm = matrix_norm(&matrix, Some("fro"))?;    // Frobenius
let one_norm = matrix_norm(&matrix, Some("1"))?;      // 1-norm
let inf_norm = matrix_norm(&matrix, Some("inf"))?;    // Infinity-norm
let two_norm = matrix_norm(&matrix, Some("2"))?;      // 2-norm (spectral)
let nuc_norm = matrix_norm(&matrix, Some("nuc"))?;    // Nuclear norm
```

### Special Matrix Construction

#### Standard Matrices
```rust
use torsh_linalg::special_matrices::{eye, diag, zeros, ones};

// Identity matrix
let identity = eye::<f32>(5)?;

// Diagonal matrix from vector
let diagonal = diag(&vector, 0)?; // Main diagonal
let super_diag = diag(&vector, 1)?; // Super-diagonal
let sub_diag = diag(&vector, -1)?; // Sub-diagonal

// Extract diagonal from matrix
let main_diag = diag(&matrix, 0)?;
```

#### Structured Matrices
```rust
use torsh_linalg::special_matrices::{
    vandermonde, toeplitz, hankel, circulant
};

// Vandermonde matrix
let vander = vandermonde(&points, Some(degree))?;

// Toeplitz matrix (constant along diagonals)
let toep = toeplitz(&column, &row)?;

// Hankel matrix (constant along anti-diagonals)
let hank = hankel(&column, &row)?;

// Circulant matrix
let circ = circulant(&first_row)?;
```

### Advanced Numerical Methods

#### Error Analysis and Refinement
```rust
use torsh_linalg::solve::{
    iterative_refinement, backward_error, forward_error_bound
};

// Iterative refinement for improved accuracy
let (x_refined, residual) = iterative_refinement(&a, &b, &x_initial, 3)?;

// Error analysis
let backward_err = backward_error(&a, &b, &x)?;
let forward_bound = forward_error_bound(&a, &backward_err)?;
```

#### Regularization Techniques
```rust
use torsh_linalg::solve::{
    tikhonov_regularization, truncated_svd_solve, damped_least_squares
};

// Ridge regression (Tikhonov regularization)
let x = tikhonov_regularization(&a, &b, 0.01)?;

// Truncated SVD for rank-deficient problems
let x = truncated_svd_solve(&a, &b, 10)?; // Keep 10 largest singular values

// Damped least squares
let x = damped_least_squares(&a, &b, 0.1, Some(&prior))?;
```

#### Stability Analysis
```rust
use torsh_linalg::{stability_analysis, cond_estimate};

// Comprehensive stability analysis
let (condition_num, rank_strict, rank_numerical, stability_metric) = 
    stability_analysis(&matrix)?;

// Efficient condition number estimation
let cond_est = cond_estimate(&matrix, Some("2"), Some(100))?;
```

### Einsum Operations

ToRSh-linalg supports Einstein summation notation for tensor contractions:

```rust
use torsh_linalg::einsum;

// Matrix multiplication
let c = einsum("ij,jk->ik", &[&a, &b])?;

// Batch matrix multiplication
let c = einsum("bij,bjk->bik", &[&a, &b])?;

// Matrix transpose
let at = einsum("ij->ji", &[&a])?;

// Trace computation
let trace = einsum("ii->", &[&a])?;

// Outer product
let outer = einsum("i,j->ij", &[&u, &v])?;

// Inner product
let inner = einsum("i,i->", &[&u, &v])?;

// Diagonal extraction
let diag = einsum("ii->i", &[&a])?;
```

## Performance Considerations

### Memory Layout Optimization
- Use contiguous tensors when possible
- Consider tensor transposition costs
- Batch operations for better cache utilization

### Algorithm Selection
- Small matrices: Direct methods (LU, Cholesky)
- Large sparse matrices: Iterative methods (CG, GMRES)
- Structured matrices: Specialized algorithms (Thomas, FFT-based)

### Numerical Stability
- Use pivoting for LU decomposition
- Apply iterative refinement for critical accuracy
- Monitor condition numbers for ill-conditioned systems

## Error Handling

All operations return `Result<T>` types and provide detailed error messages:

```rust
match solve(&a, &b) {
    Ok(solution) => println!("Solution found"),
    Err(TorshError::SingularMatrix(msg)) => {
        println!("Matrix is singular: {}", msg);
    },
    Err(TorshError::InvalidArgument(msg)) => {
        println!("Invalid input: {}", msg);
    },
    Err(e) => println!("Other error: {:?}", e),
}
```

## Best Practices

1. **Check matrix conditions**: Always verify matrix properties before solving
2. **Use appropriate solvers**: Match the solver to your problem structure
3. **Handle numerical precision**: Use appropriate tolerances for your application
4. **Monitor convergence**: Check iteration counts and residuals for iterative methods
5. **Validate results**: Compute residuals and error estimates when accuracy is critical