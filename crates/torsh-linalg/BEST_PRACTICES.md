# Best Practices for ToRSh Linear Algebra

## Overview

This guide provides best practices, recommendations, and common pitfalls to avoid when using torsh-linalg for numerical linear algebra computations. Following these guidelines will help you write robust, efficient, and numerically stable code.

## Algorithm Selection

### Choose the Right Algorithm for Your Problem

#### Matrix Structure-Aware Selection
```rust
// ✅ Good: Exploit structure
fn solve_system(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if is_symmetric_positive_definite(a)? {
        // Use Cholesky: 2x faster than LU for SPD matrices
        cholesky_solve(a, b)
    } else if is_tridiagonal(a)? {
        // Use Thomas algorithm: O(n) vs O(n³)
        thomas_algorithm_solve(a, b)
    } else {
        // General case: LU with pivoting
        solve(a, b)
    }
}

// ❌ Bad: Always use general algorithm
fn solve_system_bad(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    solve(a, b) // Misses optimization opportunities
}
```

#### Size-Based Algorithm Selection
```rust
fn optimal_matrix_multiply(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let n = a.shape().dims()[0];
    
    match n {
        0..=3 => {
            // Use direct formulas for very small matrices
            direct_multiply(a, b)
        },
        4..=100 => {
            // Standard algorithm for small-medium matrices
            matmul(a, b)
        },
        101..=1000 => {
            // Blocked algorithm for better cache performance
            blocked_matmul(a, b)
        },
        _ => {
            // Consider parallel or out-of-core algorithms
            parallel_matmul(a, b)
        }
    }
}
```

### Condition Number Awareness

```rust
// ✅ Good: Check condition before solving
fn robust_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let condition_num = cond(a, Some("2"))?;
    
    match condition_num {
        c if c < 1e3 => {
            // Well-conditioned: use direct method
            solve(a, b)
        },
        c if c < 1e12 => {
            // Moderately ill-conditioned: use iterative refinement
            let x = solve(a, b)?;
            iterative_refinement(a, b, &x, 2)
        },
        _ => {
            // Severely ill-conditioned: use regularization
            println!("Warning: Matrix is ill-conditioned (κ = {:.2e})", condition_num);
            tikhonov_regularization(a, b, 1e-8)
        }
    }
}

// ❌ Bad: Ignore conditioning
fn naive_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    solve(a, b) // May fail or give inaccurate results
}
```

## Numerical Stability Guidelines

### Use Stable Algorithms

```rust
// ✅ Good: Use pivoted decompositions
fn stable_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // LU with partial pivoting is backward stable
    let (p, l, u) = lu(a)?;
    let pb = matvec(&p, b)?;
    let y = solve_triangular(&l, &pb, false, false)?;
    solve_triangular(&u, &y, true, false)
}

// ❌ Bad: Use unpivoted decomposition for general matrices
fn unstable_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Gaussian elimination without pivoting can be unstable
    naive_gaussian_elimination(a, b)
}
```

### Tolerance Selection

```rust
// ✅ Good: Use appropriate tolerances
const MACHINE_EPS_F32: f32 = 1.2e-7;
const SQRT_EPS_F32: f32 = 3.5e-4;

fn set_tolerances(problem_size: usize, condition_num: f32) -> (f32, f32) {
    // Scale tolerances based on problem characteristics
    let base_tol = 100.0 * MACHINE_EPS_F32;
    let solve_tolerance = base_tol * condition_num.sqrt();
    let rank_tolerance = SQRT_EPS_F32 * (problem_size as f32).sqrt();
    
    (solve_tolerance, rank_tolerance)
}

// ❌ Bad: Use fixed tolerances everywhere
const FIXED_TOL: f32 = 1e-6; // May be too strict or too loose
```

### Avoid Catastrophic Cancellation

```rust
// ✅ Good: Reformulate to avoid cancellation
fn stable_quadratic_formula(a: f32, b: f32, c: f32) -> (f32, f32) {
    let discriminant = b * b - 4.0 * a * c;
    let sqrt_d = discriminant.sqrt();
    
    // Avoid cancellation by using different formulas for each root
    let (x1, x2) = if b >= 0.0 {
        let x1 = (-b - sqrt_d) / (2.0 * a);
        let x2 = (2.0 * c) / (-b - sqrt_d); // Alternative formula
        (x1, x2)
    } else {
        let x1 = (2.0 * c) / (-b + sqrt_d); // Alternative formula
        let x2 = (-b + sqrt_d) / (2.0 * a);
        (x1, x2)
    };
    
    (x1, x2)
}

// ❌ Bad: Prone to cancellation
fn unstable_quadratic_formula(a: f32, b: f32, c: f32) -> (f32, f32) {
    let discriminant = b * b - 4.0 * a * c;
    let sqrt_d = discriminant.sqrt();
    
    // Both formulas can suffer from cancellation
    let x1 = (-b + sqrt_d) / (2.0 * a);
    let x2 = (-b - sqrt_d) / (2.0 * a);
    
    (x1, x2)
}
```

## Memory Management

### Minimize Allocations in Hot Paths

```rust
// ✅ Good: Reuse workspace
struct LinearSolver {
    workspace: Option<Tensor>,
    factorization_cache: Option<(Tensor, Tensor, Tensor)>,
}

impl LinearSolver {
    fn solve_with_cache(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Reuse workspace if size matches
        if let Some(ref ws) = self.workspace {
            if ws.shape() == b.shape() {
                ws.copy_from(b)?;
                return self.solve_inplace(a, ws);
            }
        }
        
        // Allocate new workspace if needed
        self.workspace = Some(b.clone());
        self.solve_inplace(a, self.workspace.as_ref().unwrap())
    }
}

// ❌ Bad: Allocate on every call
fn solve_with_allocation(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let workspace = b.clone(); // New allocation every time
    solve(a, &workspace)
}
```

### Use In-Place Operations When Possible

```rust
// ✅ Good: In-place operations
fn efficient_matrix_operations(a: &mut Tensor, b: &Tensor) -> Result<()> {
    a.add_assign(b)?;           // In-place addition
    a.mul_scalar_assign(2.0)?;  // In-place scalar multiplication
    a.transpose_inplace()?;     // In-place transpose (if possible)
    Ok(())
}

// ❌ Bad: Unnecessary copies
fn inefficient_matrix_operations(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let temp1 = a.add(b)?;      // Creates new tensor
    let temp2 = temp1.mul_scalar(2.0)?; // Creates another tensor
    let result = temp2.transpose(-2, -1)?; // Yet another tensor
    Ok(result) // temp1 and temp2 are wasted
}
```

### Memory Layout Considerations

```rust
// ✅ Good: Consider access patterns
fn cache_friendly_iteration(matrix: &Tensor) -> Result<f32> {
    let shape = matrix.shape();
    let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
    
    let mut sum = 0.0f32;
    
    // Row-major access for row-major storage
    for i in 0..rows {
        for j in 0..cols {
            sum += matrix.get(&[i, j])?;
        }
    }
    
    Ok(sum)
}

// ❌ Bad: Poor cache locality
fn cache_unfriendly_iteration(matrix: &Tensor) -> Result<f32> {
    let shape = matrix.shape();
    let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
    
    let mut sum = 0.0f32;
    
    // Column-major access for row-major storage
    for j in 0..cols {
        for i in 0..rows {
            sum += matrix.get(&[i, j])?; // Poor cache locality
        }
    }
    
    Ok(sum)
}
```

## Error Handling and Validation

### Comprehensive Input Validation

```rust
// ✅ Good: Thorough validation
fn validate_and_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check dimensions
    if a.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be 2-dimensional".to_string()
        ));
    }
    
    if b.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Vector b must be 1-dimensional".to_string()
        ));
    }
    
    let (n, m) = (a.shape().dims()[0], a.shape().dims()[1]);
    let b_len = b.shape().dims()[0];
    
    if n != m {
        return Err(TorshError::InvalidArgument(
            format!("Matrix A must be square, got {}x{}", n, m)
        ));
    }
    
    if n != b_len {
        return Err(TorshError::InvalidArgument(
            format!("Incompatible dimensions: A is {}x{}, b has length {}", n, m, b_len)
        ));
    }
    
    // Check for NaN or infinite values
    if has_nan_or_inf(a)? || has_nan_or_inf(b)? {
        return Err(TorshError::InvalidArgument(
            "Input contains NaN or infinite values".to_string()
        ));
    }
    
    solve(a, b)
}

// ❌ Bad: No validation
fn unsafe_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    solve(a, b) // Will panic or give wrong results on bad input
}
```

### Graceful Degradation

```rust
// ✅ Good: Multiple fallback strategies
fn robust_matrix_inverse(a: &Tensor) -> Result<Tensor> {
    // Try Cholesky first (fastest for SPD matrices)
    if let Ok(inv) = cholesky_inverse(a) {
        return Ok(inv);
    }
    
    // Try LU decomposition
    if let Ok(inv) = lu_inverse(a) {
        return Ok(inv);
    }
    
    // Fall back to SVD-based pseudoinverse
    println!("Warning: Matrix may be singular, computing pseudoinverse");
    pinv(a, Some(1e-12))
}

// ❌ Bad: Single method, no fallback
fn fragile_inverse(a: &Tensor) -> Result<Tensor> {
    inv(a) // Fails completely if matrix is singular
}
```

## Performance Optimization

### Profile Before Optimizing

```rust
use std::time::Instant;

// ✅ Good: Measure performance systematically
fn benchmark_algorithms(matrix_sizes: &[usize]) {
    println!("Algorithm Comparison:");
    println!("Size\tDirect(ms)\tIterative(ms)\tSpeedup");
    
    for &n in matrix_sizes {
        let a = create_test_matrix(n);
        let b = create_test_vector(n);
        
        // Benchmark direct method
        let start = Instant::now();
        let _x1 = solve(&a, &b).unwrap();
        let direct_time = start.elapsed().as_millis();
        
        // Benchmark iterative method
        let start = Instant::now();
        let _x2 = cg(&a, &b, None, Some(1e-8), Some(1000)).unwrap();
        let iterative_time = start.elapsed().as_millis();
        
        let speedup = direct_time as f64 / iterative_time as f64;
        println!("{}\t{}\t\t{}\t\t{:.2}x", n, direct_time, iterative_time, speedup);
    }
}
```

### Use Problem-Specific Optimizations

```rust
// ✅ Good: Exploit problem structure
fn solve_banded_system(a: &BandedMatrix, b: &Tensor) -> Result<Tensor> {
    match a.bandwidth() {
        1 => thomas_algorithm(a, b),        // Tridiagonal
        2 => pentadiagonal_solve(a, b),     // Pentadiagonal
        _ => general_banded_solve(a, b),    // General banded
    }
}

// ✅ Good: Batch operations for efficiency
fn solve_multiple_systems(a: &Tensor, b_vectors: &[&Tensor]) -> Result<Vec<Tensor>> {
    // Factor once, solve many
    let (p, l, u) = lu(a)?;
    
    let mut solutions = Vec::with_capacity(b_vectors.len());
    for &b in b_vectors {
        let pb = matvec(&p, b)?;
        let y = solve_triangular(&l, &pb, false, false)?;
        let x = solve_triangular(&u, &y, true, false)?;
        solutions.push(x);
    }
    
    Ok(solutions)
}
```

### Avoid Premature Optimization

```rust
// ✅ Good: Start with clear, correct code
fn clear_implementation(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Clear, readable implementation first
    let condition_num = cond(a, Some("2"))?;
    
    if condition_num < 1e12 {
        solve(a, b)
    } else {
        tikhonov_regularization(a, b, 1e-8)
    }
}

// Only optimize after profiling shows this is a bottleneck
fn optimized_implementation(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // ... more complex optimized code
}
```

## Testing and Validation

### Comprehensive Test Coverage

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    // ✅ Good: Test multiple scenarios
    #[test]
    fn test_solve_comprehensive() -> Result<()> {
        // Test well-conditioned system
        test_well_conditioned_system()?;
        
        // Test ill-conditioned system
        test_ill_conditioned_system()?;
        
        // Test singular system
        test_singular_system()?;
        
        // Test edge cases
        test_edge_cases()?;
        
        Ok(())
    }
    
    fn test_well_conditioned_system() -> Result<()> {
        let a = create_well_conditioned_matrix(10);
        let x_true = create_test_vector(10);
        let b = matvec(&a, &x_true)?;
        
        let x_computed = solve(&a, &b)?;
        
        // Check relative error
        let error = x_true.sub(&x_computed)?;
        let relative_error = error.norm(None)?.get(&[])? / x_true.norm(None)?.get(&[])?;
        
        assert!(relative_error < 1e-10, "Solution error too large: {}", relative_error);
        
        Ok(())
    }
    
    fn test_ill_conditioned_system() -> Result<()> {
        let a = create_hilbert_matrix(8); // Known ill-conditioned matrix
        let x_true = create_test_vector(8);
        let b = matvec(&a, &x_true)?;
        
        let x_computed = solve(&a, &b)?;
        
        // For ill-conditioned systems, expect larger errors
        let condition_num = cond(&a, Some("2"))?;
        let expected_error = condition_num * 1e-15; // Machine precision scaled by condition
        
        let error = x_true.sub(&x_computed)?;
        let relative_error = error.norm(None)?.get(&[])? / x_true.norm(None)?.get(&[])?;
        
        assert!(relative_error < expected_error, 
                "Error {} exceeds expected bound {}", relative_error, expected_error);
        
        Ok(())
    }
}
```

### Property-Based Testing

```rust
// ✅ Good: Test mathematical properties
#[test]
fn test_decomposition_properties() -> Result<()> {
    let a = create_random_matrix(5, 5);
    
    // Test LU decomposition: PA = LU
    let (p, l, u) = lu(&a)?;
    let pa = matmul(&p, &a)?;
    let lu_product = matmul(&l, &u)?;
    
    let error = pa.sub(&lu_product)?;
    let error_norm = matrix_norm(&error, Some("fro"))?;
    
    assert!(error_norm < 1e-10, "LU decomposition property violated");
    
    // Test QR decomposition: A = QR
    let (q, r) = qr(&a)?;
    let qr_product = matmul(&q, &r)?;
    
    let error = a.sub(&qr_product)?;
    let error_norm = matrix_norm(&error, Some("fro"))?;
    
    assert!(error_norm < 1e-10, "QR decomposition property violated");
    
    // Test Q is orthogonal: Q^T Q = I
    let qt = q.transpose(-2, -1)?;
    let qtq = matmul(&qt, &q)?;
    let identity = eye::<f32>(5);
    
    let error = qtq.sub(&identity)?;
    let error_norm = matrix_norm(&error, Some("fro"))?;
    
    assert!(error_norm < 1e-6, "Q matrix not orthogonal");
    
    Ok(())
}
```

## Documentation and Maintainability

### Clear Function Documentation

```rust
/// Solves the linear system Ax = b using the most appropriate method
/// based on matrix properties.
///
/// # Arguments
/// * `a` - Coefficient matrix (must be square and non-singular)
/// * `b` - Right-hand side vector
///
/// # Returns
/// * `Ok(x)` - Solution vector
/// * `Err(TorshError)` - If matrix is singular or inputs are invalid
///
/// # Examples
/// ```rust
/// use torsh_linalg::solve;
/// use torsh_tensor::Tensor;
///
/// let a = Tensor::from_data(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2], DeviceType::Cpu);
/// let b = Tensor::from_data(vec![3.0, 3.0], vec![2], DeviceType::Cpu);
/// let x = solve(&a, &b)?;
/// ```
///
/// # Numerical Notes
/// - Uses LU decomposition with partial pivoting for stability
/// - Condition number is checked; regularization applied if needed
/// - Complexity: O(n³) for n×n matrix
pub fn adaptive_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Implementation...
}
```

### Code Organization

```rust
// ✅ Good: Organized module structure
pub mod decomposition {
    //! Matrix decomposition algorithms
    pub mod lu;
    pub mod qr;
    pub mod svd;
    pub mod eigen;
}

pub mod solve {
    //! Linear system solvers
    pub mod direct;
    pub mod iterative;
    pub mod specialized;
}

pub mod utils {
    //! Utility functions for validation and error handling
    pub mod validation;
    pub mod error_analysis;
}
```

## Common Pitfalls to Avoid

### Don't Ignore Numerical Issues

```rust
// ❌ Bad: Ignore warning signs
fn dangerous_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let x = solve(a, b)?;
    Ok(x) // No verification of result quality
}

// ✅ Good: Validate results
fn safe_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let x = solve(a, b)?;
    
    // Check residual
    let residual = matvec(a, &x)?.sub(b)?;
    let residual_norm = residual.norm(None)?.get(&[])?;
    let b_norm = b.norm(None)?.get(&[])?;
    let relative_residual = residual_norm / b_norm;
    
    if relative_residual > 1e-8 {
        println!("Warning: Large residual norm: {:.2e}", relative_residual);
    }
    
    Ok(x)
}
```

### Don't Use Wrong Data Types

```rust
// ❌ Bad: Using wrong precision
fn low_precision_computation() -> Result<f32> {
    let a = create_large_matrix_f32(1000); // f32 for large computation
    let condition_num = cond(&a, Some("2"))?;
    
    if condition_num > 1e6 {
        // f32 precision insufficient for this condition number
        println!("Warning: Precision may be inadequate");
    }
    
    Ok(condition_num)
}

// ✅ Good: Use appropriate precision
fn appropriate_precision_computation() -> Result<f64> {
    let a = create_large_matrix_f64(1000); // f64 for high-precision needs
    let condition_num = cond(&a, Some("2"))?;
    
    Ok(condition_num)
}
```

### Don't Assume Algorithm Convergence

```rust
// ❌ Bad: Assume iterative methods always converge
fn naive_iterative_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    cg(a, b, None, Some(1e-8), Some(1000)) // May not converge
}

// ✅ Good: Handle convergence failure
fn robust_iterative_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    match cg(a, b, None, Some(1e-8), Some(1000)) {
        Ok(solution) => Ok(solution),
        Err(_) => {
            println!("Iterative method failed to converge, trying direct method");
            solve(a, b)
        }
    }
}
```

## Summary

### Key Principles

1. **Algorithm Selection**: Choose algorithms based on problem structure and size
2. **Numerical Stability**: Always use stable algorithms and check conditioning
3. **Error Handling**: Validate inputs and handle failures gracefully
4. **Performance**: Profile before optimizing, exploit structure when possible
5. **Testing**: Test mathematical properties and edge cases thoroughly
6. **Documentation**: Clearly document assumptions and limitations

### Quick Checklist

Before deploying linear algebra code:

- [ ] Input validation implemented
- [ ] Condition number checked for linear systems
- [ ] Appropriate tolerances selected
- [ ] Error bounds computed and verified
- [ ] Memory usage optimized for problem size
- [ ] Fallback strategies implemented
- [ ] Comprehensive tests written
- [ ] Performance benchmarked against alternatives
- [ ] Documentation includes numerical considerations
- [ ] Code reviewed for numerical stability

Following these best practices will help you build robust, efficient, and maintainable numerical applications using torsh-linalg.