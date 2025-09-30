# Comprehensive Examples for ToRSh Linear Algebra

## Overview

This document provides practical examples demonstrating how to use torsh-linalg for various linear algebra applications. Examples range from basic operations to advanced numerical methods.

## Basic Operations

### Matrix Creation and Basic Operations

```rust
use torsh_linalg::*;
use torsh_tensor::{Tensor, creation::*};
use torsh_core::DeviceType;

fn basic_matrix_operations() -> Result<()> {
    // Create matrices
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_data(data_a, vec![2, 2], DeviceType::Cpu);
    
    let data_b = vec![5.0f32, 6.0, 7.0, 8.0];
    let b = Tensor::from_data(data_b, vec![2, 2], DeviceType::Cpu);
    
    // Matrix multiplication
    let c = matmul(&a, &b)?;
    println!("A * B = {:?}", c);
    
    // Matrix properties
    let det_a = det(&a)?;
    let trace_a = trace(&a)?;
    let rank_a = matrix_rank(&a, None)?;
    
    println!("det(A) = {}", det_a);
    println!("trace(A) = {}", trace_a);
    println!("rank(A) = {}", rank_a);
    
    // Matrix norms
    let fro_norm = matrix_functions::matrix_norm(&a, Some("fro"))?;
    let two_norm = matrix_functions::matrix_norm(&a, Some("2"))?;
    
    println!("||A||_F = {}", fro_norm);
    println!("||A||_2 = {}", two_norm);
    
    Ok(())
}
```

### Vector Operations

```rust
use torsh_linalg::*;

fn vector_operations() -> Result<()> {
    // Create vectors
    let u = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu);
    let v = Tensor::from_data(vec![4.0f32, 5.0, 6.0], vec![3], DeviceType::Cpu);
    
    // Inner product (dot product)
    let dot_product = inner(&u, &v)?;
    println!("u · v = {:?}", dot_product);
    
    // Outer product
    let outer_product = outer(&u, &v)?;
    println!("u ⊗ v = {:?}", outer_product);
    
    // Matrix-vector operations
    let a = eye::<f32>(3);
    let result = matvec(&a, &u)?;
    println!("A * u = {:?}", result);
    
    Ok(())
}
```

## Matrix Decompositions

### LU Decomposition Example

```rust
use torsh_linalg::decomposition::*;

fn lu_decomposition_example() -> Result<()> {
    // Create a test matrix
    let data = vec![
        2.0f32, 1.0, 1.0,
        4.0, 3.0, 3.0,
        8.0, 7.0, 9.0
    ];
    let a = Tensor::from_data(data, vec![3, 3], DeviceType::Cpu);
    
    // Perform LU decomposition
    let (p, l, u) = lu(&a)?;
    
    println!("Original matrix A:");
    print_matrix(&a);
    
    println!("Permutation matrix P:");
    print_matrix(&p);
    
    println!("Lower triangular L:");
    print_matrix(&l);
    
    println!("Upper triangular U:");
    print_matrix(&u);
    
    // Verify PA = LU
    let pa = matmul(&p, &a)?;
    let lu_product = matmul(&l, &u)?;
    
    println!("Verification ||PA - LU||:");
    let diff = pa.sub(&lu_product)?;
    let error = matrix_functions::matrix_norm(&diff, Some("fro"))?;
    println!("Error: {}", error);
    
    Ok(())
}

fn print_matrix(matrix: &Tensor) {
    let shape = matrix.shape();
    let dims = shape.dims();
    
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            if let Ok(val) = matrix.get(&[i, j]) {
                print!("{:8.4} ", val);
            }
        }
        println!();
    }
    println!();
}
```

### SVD for Data Analysis

```rust
use torsh_linalg::decomposition::*;

fn svd_data_analysis() -> Result<()> {
    // Create a data matrix (features × samples)
    let data = vec![
        1.0f32, 2.0, 3.0, 4.0,
        2.0, 4.0, 6.0, 8.0,
        1.0, 3.0, 5.0, 7.0,
        3.0, 6.0, 9.0, 12.0
    ];
    let x = Tensor::from_data(data, vec![4, 4], DeviceType::Cpu);
    
    // Perform SVD
    let (u, s, vt) = svd(&x, true)?;
    
    println!("Singular values:");
    for i in 0..s.shape().dims()[0] {
        if let Ok(val) = s.get(&[i]) {
            println!("σ_{} = {:.6}", i, val);
        }
    }
    
    // Compute effective rank
    let tolerance = 1e-6;
    let mut effective_rank = 0;
    for i in 0..s.shape().dims()[0] {
        if let Ok(val) = s.get(&[i]) {
            if val > tolerance {
                effective_rank += 1;
            }
        }
    }
    println!("Effective rank: {}", effective_rank);
    
    // Low-rank approximation
    let rank_k = 2;
    let u_k = u.slice(0, 0, rank_k)?;
    let s_k = s.slice(0, 0, rank_k)?;
    let vt_k = vt.slice(0, 0, rank_k)?;
    
    // Reconstruct approximation
    let s_diag = special_matrices::diag(&s_k, 0)?;
    let us = matmul(&u_k, &s_diag)?;
    let x_approx = matmul(&us, &vt_k)?;
    
    // Compute approximation error
    let error = x.sub(&x_approx)?;
    let approximation_error = matrix_functions::matrix_norm(&error, Some("fro"))?;
    println!("Rank-{} approximation error: {:.6}", rank_k, approximation_error);
    
    Ok(())
}
```

### Principal Component Analysis (PCA)

```rust
use torsh_linalg::*;

fn principal_component_analysis() -> Result<()> {
    // Sample data matrix (observations × features)
    let data = vec![
        1.0f32, 2.0, 3.0,
        2.0, 3.0, 4.0,
        3.0, 4.0, 5.0,
        4.0, 5.0, 6.0,
        5.0, 6.0, 7.0
    ];
    let x = Tensor::from_data(data, vec![5, 3], DeviceType::Cpu);
    
    // Center the data (subtract mean)
    let n_samples = x.shape().dims()[0] as f32;
    let mean = x.sum_dim(0, false)?.div_scalar(n_samples)?;
    let x_centered = x.sub(&mean.unsqueeze(0)?)?;
    
    // Compute covariance matrix
    let xt = x_centered.transpose(-2, -1)?;
    let cov = matmul(&xt, &x_centered)?.div_scalar(n_samples - 1.0)?;
    
    // Eigendecomposition of covariance matrix
    let (eigenvals, eigenvecs) = decomposition::eig(&cov)?;
    
    println!("Principal component analysis results:");
    println!("Eigenvalues (explained variance):");
    
    let mut total_variance = 0.0f32;
    for i in 0..eigenvals.shape().dims()[0] {
        if let Ok(val) = eigenvals.get(&[i]) {
            total_variance += val;
        }
    }
    
    for i in 0..eigenvals.shape().dims()[0] {
        if let Ok(val) = eigenvals.get(&[i]) {
            let explained_ratio = val / total_variance;
            println!("PC{}: {:.4} ({:.1}%)", i+1, val, explained_ratio * 100.0);
        }
    }
    
    // Project data onto first two principal components
    let pc_components = eigenvecs.slice(1, 0, 2)?; // First 2 components
    let x_projected = matmul(&x_centered, &pc_components)?;
    
    println!("Projected data (first 2 PCs):");
    print_matrix(&x_projected);
    
    Ok(())
}
```

## Linear System Solving

### Solving Linear Systems

```rust
use torsh_linalg::solve::*;

fn linear_system_solving() -> Result<()> {
    // System: Ax = b
    let a_data = vec![
        3.0f32, 2.0, -1.0,
        2.0, -2.0, 4.0,
        -1.0, 0.5, -1.0
    ];
    let a = Tensor::from_data(a_data, vec![3, 3], DeviceType::Cpu);
    
    let b = Tensor::from_data(vec![1.0f32, -2.0, 0.0], vec![3], DeviceType::Cpu);
    
    // Direct solve using LU decomposition
    let x = solve(&a, &b)?;
    
    println!("Solution x:");
    for i in 0..x.shape().dims()[0] {
        if let Ok(val) = x.get(&[i]) {
            println!("x[{}] = {:.6}", i, val);
        }
    }
    
    // Verify solution
    let ax = matvec(&a, &x)?;
    let residual = b.sub(&ax)?;
    let residual_norm = matrix_functions::matrix_norm(&residual.unsqueeze(1)?, Some("2"))?;
    println!("Residual norm: {:.2e}", residual_norm);
    
    // Check condition number
    let condition = cond(&a, Some("2"))?;
    println!("Condition number: {:.2e}", condition);
    
    Ok(())
}
```

### Least Squares Problems

```rust
use torsh_linalg::solve::*;

fn least_squares_example() -> Result<()> {
    // Overdetermined system: more equations than unknowns
    // Fit line y = ax + b to data points
    
    // Data points: (1,1), (2,3), (3,3), (4,6), (5,8)
    let x_coords = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let y_coords = vec![1.0f32, 3.0, 3.0, 6.0, 8.0];
    
    // Design matrix for linear regression: [x, 1]
    let mut a_data = Vec::new();
    for &x in &x_coords {
        a_data.push(x);
        a_data.push(1.0);
    }
    let a = Tensor::from_data(a_data, vec![5, 2], DeviceType::Cpu);
    let b = Tensor::from_data(y_coords, vec![5], DeviceType::Cpu);
    
    // Solve least squares problem
    let coeffs = least_squares(&a, &b)?;
    
    let slope = coeffs.get(&[0])?;
    let intercept = coeffs.get(&[1])?;
    
    println!("Linear regression results:");
    println!("y = {:.4}x + {:.4}", slope, intercept);
    
    // Compute residual
    let y_pred = matvec(&a, &coeffs)?;
    let residual = b.sub(&y_pred)?;
    let rss = residual.dot(&residual)?; // Residual sum of squares
    println!("Residual sum of squares: {:.6}", rss.get(&[])?);
    
    // R-squared calculation
    let y_mean = b.mean(None, false)?;
    let y_centered = b.sub(&y_mean)?;
    let tss = y_centered.dot(&y_centered)?; // Total sum of squares
    let r_squared = 1.0 - rss.get(&[])? / tss.get(&[])?;
    println!("R-squared: {:.4}", r_squared);
    
    Ok(())
}
```

### Iterative Solvers for Large Systems

```rust
use torsh_linalg::sparse::*;

fn iterative_solver_example() -> Result<()> {
    // Create a symmetric positive definite system
    // A = D + S + S^T where D is diagonal and S is strictly upper triangular
    
    let n = 100;
    let mut a_data = vec![0.0f32; n * n];
    
    // Fill diagonal
    for i in 0..n {
        a_data[i * n + i] = 4.0; // Diagonal dominance
    }
    
    // Fill off-diagonals (tridiagonal pattern)
    for i in 0..n-1 {
        a_data[i * n + (i + 1)] = -1.0; // Super-diagonal
        a_data[(i + 1) * n + i] = -1.0; // Sub-diagonal
    }
    
    let a = Tensor::from_data(a_data, vec![n, n], DeviceType::Cpu);
    
    // Right-hand side (solution should be all ones)
    let mut b_data = vec![2.0f32; n];
    b_data[0] = 3.0;  // Boundary condition
    b_data[n-1] = 3.0; // Boundary condition
    let b = Tensor::from_data(b_data, vec![n], DeviceType::Cpu);
    
    // Solve using Conjugate Gradient
    println!("Solving {}x{} system with CG...", n, n);
    
    let tolerance = 1e-8;
    let max_iterations = 1000;
    
    let x = cg(&a, &b, None, Some(tolerance), Some(max_iterations))?;
    
    // Verify solution
    let ax = matvec(&a, &x)?;
    let residual = b.sub(&ax)?;
    let residual_norm = residual.norm(None)?;
    
    println!("Final residual norm: {:.2e}", residual_norm.get(&[])?);
    
    // With diagonal preconditioning
    println!("Solving with diagonal preconditioning...");
    let preconditioner = DiagonalPreconditioner::new(&a)?;
    let x_precond = cg(&a, &b, Some(&preconditioner), Some(tolerance), Some(max_iterations))?;
    
    let ax_precond = matvec(&a, &x_precond)?;
    let residual_precond = b.sub(&ax_precond)?;
    let residual_norm_precond = residual_precond.norm(None)?;
    
    println!("Preconditioned residual norm: {:.2e}", residual_norm_precond.get(&[])?);
    
    Ok(())
}
```

## Advanced Applications

### Eigenvalue Problems

```rust
use torsh_linalg::decomposition::*;

fn eigenvalue_problems() -> Result<()> {
    // Create a symmetric matrix (guarantees real eigenvalues)
    let data = vec![
        4.0f32, -2.0, 1.0,
        -2.0, 2.0, -4.0,
        1.0, -4.0, 6.0
    ];
    let a = Tensor::from_data(data, vec![3, 3], DeviceType::Cpu);
    
    // Compute all eigenvalues and eigenvectors
    let (eigenvals, eigenvecs) = eig(&a)?;
    
    println!("Eigenvalue decomposition:");
    for i in 0..eigenvals.shape().dims()[0] {
        if let Ok(lambda) = eigenvals.get(&[i]) {
            println!("λ_{} = {:.6}", i, lambda);
            
            // Extract corresponding eigenvector
            let mut eigenvec = Vec::new();
            for j in 0..eigenvecs.shape().dims()[0] {
                if let Ok(val) = eigenvecs.get(&[j, i]) {
                    eigenvec.push(val);
                }
            }
            println!("  v_{} = [{:.4}, {:.4}, {:.4}]", i, eigenvec[0], eigenvec[1], eigenvec[2]);
        }
    }
    
    // Verify eigenvalue equation: Av = λv
    for i in 0..eigenvals.shape().dims()[0] {
        if let Ok(lambda) = eigenvals.get(&[i]) {
            // Extract eigenvector
            let v_data = (0..eigenvecs.shape().dims()[0])
                .map(|j| eigenvecs.get(&[j, i]).unwrap_or(0.0))
                .collect();
            let v = Tensor::from_data(v_data, vec![3], DeviceType::Cpu);
            
            // Compute Av
            let av = matvec(&a, &v)?;
            
            // Compute λv
            let lambda_v = v.mul_scalar(lambda)?;
            
            // Check ||Av - λv||
            let diff = av.sub(&lambda_v)?;
            let error = diff.norm(None)?.get(&[])?;
            println!("  Verification error for λ_{}: {:.2e}", i, error);
        }
    }
    
    Ok(())
}
```

### Matrix Functions Application

```rust
use torsh_linalg::matrix_functions::*;

fn matrix_functions_example() -> Result<()> {
    // Create a positive definite matrix
    let data = vec![
        2.0f32, -1.0, 0.0,
        -1.0, 2.0, -1.0,
        0.0, -1.0, 2.0
    ];
    let a = Tensor::from_data(data, vec![3, 3], DeviceType::Cpu);
    
    // Matrix square root
    let sqrt_a = matrix_sqrt(&a)?;
    println!("Matrix square root computed");
    
    // Verify: sqrt(A) * sqrt(A) = A
    let sqrt_squared = matmul(&sqrt_a, &sqrt_a)?;
    let error = a.sub(&sqrt_squared)?;
    let sqrt_error = matrix_norm(&error, Some("fro"))?;
    println!("Square root verification error: {:.2e}", sqrt_error);
    
    // Matrix exponential (useful for solving ODEs)
    let exp_a = matrix_exp(&a)?;
    println!("Matrix exponential computed");
    
    // Matrix logarithm
    let log_exp_a = matrix_log(&exp_a)?;
    let log_error = a.sub(&log_exp_a)?;
    let log_verification_error = matrix_norm(&log_error, Some("fro"))?;
    println!("Log-exp verification error: {:.2e}", log_verification_error);
    
    // Matrix power
    let a_pow_half = matrix_power(&a, 0.5)?;
    let power_error = sqrt_a.sub(&a_pow_half)?;
    let power_verification_error = matrix_norm(&power_error, Some("fro"))?;
    println!("Power(0.5) vs sqrt verification error: {:.2e}", power_verification_error);
    
    Ok(())
}
```

### Numerical Stability Analysis

```rust
use torsh_linalg::*;

fn stability_analysis_example() -> Result<()> {
    // Test different types of matrices for numerical stability
    
    // Well-conditioned matrix
    let well_conditioned = eye::<f32>(3);
    let (cond1, rank1, num_rank1, stability1) = stability_analysis(&well_conditioned)?;
    println!("Well-conditioned matrix:");
    println!("  Condition number: {:.2e}", cond1);
    println!("  Rank (strict): {}", rank1);
    println!("  Rank (numerical): {}", num_rank1);
    println!("  Stability metric: {:.4}", stability1);
    
    // Ill-conditioned matrix (Hilbert matrix)
    let n = 4;
    let mut hilbert_data = Vec::new();
    for i in 0..n {
        for j in 0..n {
            hilbert_data.push(1.0 / (i + j + 1) as f32);
        }
    }
    let hilbert = Tensor::from_data(hilbert_data, vec![n, n], DeviceType::Cpu);
    
    let (cond2, rank2, num_rank2, stability2) = stability_analysis(&hilbert)?;
    println!("\nIll-conditioned matrix (Hilbert):");
    println!("  Condition number: {:.2e}", cond2);
    println!("  Rank (strict): {}", rank2);
    println!("  Rank (numerical): {}", num_rank2);
    println!("  Stability metric: {:.4}", stability2);
    
    // Condition number estimation (faster for large matrices)
    let cond_estimate = cond_estimate(&hilbert, Some("2"), Some(50))?;
    println!("  Condition estimate: {:.2e}", cond_estimate);
    
    Ok(())
}
```

### Regularization Techniques

```rust
use torsh_linalg::solve::*;

fn regularization_example() -> Result<()> {
    // Create an ill-conditioned least squares problem
    let m = 20; // Number of observations
    let n = 15; // Number of parameters
    
    // Create design matrix with high collinearity
    let mut a_data = Vec::new();
    for i in 0..m {
        for j in 0..n {
            let val = (i as f32 * j as f32 / 10.0).sin() + 0.1 * (j as f32);
            a_data.push(val);
        }
    }
    let a = Tensor::from_data(a_data, vec![m, n], DeviceType::Cpu);
    
    // Create noisy observations
    let mut b_data = Vec::new();
    for i in 0..m {
        let true_val = (i as f32 / 2.0).sin();
        let noise = 0.1 * ((i * 7) as f32).sin(); // Simulated noise
        b_data.push(true_val + noise);
    }
    let b = Tensor::from_data(b_data, vec![m], DeviceType::Cpu);
    
    // Standard least squares (may be unstable)
    println!("Standard least squares:");
    let x_standard = least_squares(&a, &b)?;
    let residual_standard = matvec(&a, &x_standard)?.sub(&b)?;
    let rss_standard = residual_standard.norm(None)?.get(&[])?;
    println!("  Residual norm: {:.6}", rss_standard);
    
    // Tikhonov regularization (Ridge regression)
    println!("\nTikhonov regularization:");
    let lambda = 0.01;
    let x_tikhonov = tikhonov_regularization(&a, &b, lambda)?;
    let residual_tikhonov = matvec(&a, &x_tikhonov)?.sub(&b)?;
    let rss_tikhonov = residual_tikhonov.norm(None)?.get(&[])?;
    println!("  Residual norm: {:.6}", rss_tikhonov);
    println!("  Solution norm: {:.6}", x_tikhonov.norm(None)?.get(&[])?);
    
    // Truncated SVD
    println!("\nTruncated SVD:");
    let rank = 10; // Use only 10 largest singular values
    let x_tsvd = truncated_svd_solve(&a, &b, rank)?;
    let residual_tsvd = matvec(&a, &x_tsvd)?.sub(&b)?;
    let rss_tsvd = residual_tsvd.norm(None)?.get(&[])?;
    println!("  Residual norm: {:.6}", rss_tsvd);
    println!("  Effective rank: {}", rank);
    
    Ok(())
}
```

### Performance Benchmarking

```rust
use std::time::Instant;
use torsh_linalg::*;

fn performance_benchmark() -> Result<()> {
    let sizes = vec![50, 100, 200, 500];
    
    println!("Performance Benchmark Results:");
    println!("Size\tMatMul(ms)\tLU(ms)\t\tQR(ms)\t\tSVD(ms)");
    println!("-" * 60);
    
    for &n in &sizes {
        // Create random matrices
        let mut a_data = Vec::new();
        let mut b_data = Vec::new();
        
        for i in 0..n*n {
            a_data.push(((i * 17 + 13) % 100) as f32 / 100.0);
            b_data.push(((i * 23 + 7) % 100) as f32 / 100.0);
        }
        
        let a = Tensor::from_data(a_data, vec![n, n], DeviceType::Cpu);
        let b = Tensor::from_data(b_data, vec![n, n], DeviceType::Cpu);
        
        // Matrix multiplication benchmark
        let start = Instant::now();
        let _c = matmul(&a, &b)?;
        let matmul_time = start.elapsed().as_millis();
        
        // LU decomposition benchmark
        let start = Instant::now();
        let _lu_result = decomposition::lu(&a)?;
        let lu_time = start.elapsed().as_millis();
        
        // QR decomposition benchmark
        let start = Instant::now();
        let _qr_result = decomposition::qr(&a)?;
        let qr_time = start.elapsed().as_millis();
        
        // SVD benchmark
        let start = Instant::now();
        let _svd_result = decomposition::svd(&a, false)?;
        let svd_time = start.elapsed().as_millis();
        
        println!("{}\t{}\t\t{}\t\t{}\t\t{}", 
                n, matmul_time, lu_time, qr_time, svd_time);
    }
    
    Ok(())
}
```

## Utility Functions

### Error Handling Patterns

```rust
use torsh_linalg::*;
use torsh_core::TorshError;

fn robust_linear_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check matrix properties before solving
    let condition = cond(a, Some("2"))?;
    
    if condition > 1e12 {
        println!("Warning: Matrix is ill-conditioned (κ = {:.2e})", condition);
        
        // Use regularization for ill-conditioned systems
        let lambda = 1e-8 * condition; // Adaptive regularization
        return tikhonov_regularization(a, b, lambda);
    }
    
    // Check if matrix is symmetric positive definite
    if is_symmetric_positive_definite(a)? {
        // Use more efficient Cholesky-based solver
        let l = decomposition::cholesky(a, false)?;
        return solve_triangular(&l, b, false, false)?
            .and_then(|y| solve_triangular(&l.transpose(-2, -1)?, &y, true, false));
    }
    
    // Fall back to general LU-based solver
    solve(a, b)
}

fn is_symmetric_positive_definite(a: &Tensor) -> Result<bool> {
    // Check if matrix is square
    let shape = a.shape();
    let dims = shape.dims();
    if dims.len() != 2 || dims[0] != dims[1] {
        return Ok(false);
    }
    
    // Check symmetry (simplified check)
    let at = a.transpose(-2, -1)?;
    let diff = a.sub(&at)?;
    let sym_error = matrix_functions::matrix_norm(&diff, Some("fro"))?;
    
    if sym_error > 1e-10 {
        return Ok(false);
    }
    
    // Try Cholesky decomposition to check positive definiteness
    match decomposition::cholesky(a, false) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

// Example usage with comprehensive error handling
fn example_with_error_handling() -> Result<()> {
    let a_data = vec![1.0f32, 2.0, 2.0, 1.0]; // Singular matrix
    let a = Tensor::from_data(a_data, vec![2, 2], DeviceType::Cpu);
    let b = Tensor::from_data(vec![1.0f32, 1.0], vec![2], DeviceType::Cpu);
    
    match robust_linear_solve(&a, &b) {
        Ok(solution) => {
            println!("Solution found:");
            for i in 0..solution.shape().dims()[0] {
                if let Ok(val) = solution.get(&[i]) {
                    println!("x[{}] = {:.6}", i, val);
                }
            }
        },
        Err(TorshError::SingularMatrix(msg)) => {
            println!("Cannot solve: {}", msg);
            println!("Trying least squares solution...");
            let ls_solution = least_squares(&a, &b)?;
            println!("Least squares solution computed");
        },
        Err(e) => {
            println!("Unexpected error: {:?}", e);
        }
    }
    
    Ok(())
}
```

### Main Function Template

```rust
fn main() -> Result<()> {
    println!("ToRSh Linear Algebra Examples");
    println!("=" * 50);
    
    // Run basic examples
    println!("\n1. Basic Matrix Operations:");
    basic_matrix_operations()?;
    
    println!("\n2. Vector Operations:");
    vector_operations()?;
    
    println!("\n3. LU Decomposition:");
    lu_decomposition_example()?;
    
    println!("\n4. SVD Data Analysis:");
    svd_data_analysis()?;
    
    println!("\n5. Principal Component Analysis:");
    principal_component_analysis()?;
    
    println!("\n6. Linear System Solving:");
    linear_system_solving()?;
    
    println!("\n7. Least Squares:");
    least_squares_example()?;
    
    println!("\n8. Iterative Solvers:");
    iterative_solver_example()?;
    
    println!("\n9. Eigenvalue Problems:");
    eigenvalue_problems()?;
    
    println!("\n10. Matrix Functions:");
    matrix_functions_example()?;
    
    println!("\n11. Stability Analysis:");
    stability_analysis_example()?;
    
    println!("\n12. Regularization:");
    regularization_example()?;
    
    println!("\n13. Performance Benchmark:");
    performance_benchmark()?;
    
    println!("\n14. Error Handling:");
    example_with_error_handling()?;
    
    println!("\nAll examples completed successfully!");
    
    Ok(())
}
```

This comprehensive example collection demonstrates the full capabilities of torsh-linalg across various numerical linear algebra applications, from basic operations to advanced techniques used in scientific computing and machine learning.