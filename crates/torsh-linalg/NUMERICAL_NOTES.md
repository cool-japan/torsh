# Numerical Notes for ToRSh Linear Algebra

## Overview

This document provides mathematical background and numerical considerations for the linear algebra operations in torsh-linalg. Understanding these concepts is crucial for robust numerical computing.

## Fundamental Concepts

### Matrix Conditioning

#### Condition Number
The condition number κ(A) measures how sensitive a linear system Ax = b is to perturbations:

- **Definition**: κ(A) = ||A|| · ||A⁻¹||
- **Interpretation**: 
  - κ(A) = 1: Perfectly conditioned (only for orthogonal matrices)
  - κ(A) ≈ 10^k: Expect to lose about k digits of precision
  - κ(A) = ∞: Singular matrix

#### Practical Guidelines
```
κ(A) < 10³     → Well-conditioned
κ(A) ∈ [10³,10⁶] → Moderately conditioned  
κ(A) ∈ [10⁶,10¹²] → Ill-conditioned
κ(A) > 10¹²    → Effectively singular (for float32)
```

### Numerical Stability

#### Forward vs Backward Error
- **Forward Error**: ||x̃ - x|| / ||x||
- **Backward Error**: min{||ΔA|| / ||A|| : (A + ΔA)x̃ = b}
- **Stability**: A stable algorithm produces small backward error

#### Wilkinson's Principle
A numerically stable algorithm produces the exact solution to a nearby problem.

## Matrix Decompositions

### LU Decomposition

#### Mathematical Foundation
For an n×n matrix A, find permutation P, lower triangular L, and upper triangular U such that:
PA = LU

#### Numerical Considerations
- **Partial Pivoting**: Essential for stability
- **Growth Factor**: ρ = max|u_ij| / max|a_ij|
- **Stability**: LU with partial pivoting is backward stable
- **Cost**: O(2n³/3) operations

#### Implementation Notes
```rust
// Pivot selection criterion: |a_kk| = max|a_ik| for i ≥ k
// This minimizes the growth factor in practice
```

### QR Decomposition

#### Mathematical Foundation
A = QR where Q is orthogonal (Q^T Q = I) and R is upper triangular.

#### Algorithms
1. **Gram-Schmidt**: Simple but potentially unstable
2. **Modified Gram-Schmidt**: More stable
3. **Householder**: Most stable, used in LAPACK

#### Numerical Properties
- **Stability**: Householder QR is backward stable
- **Condition**: κ(R) = κ(A) for full-rank matrices
- **Cost**: O(2n³/3) operations

### Singular Value Decomposition (SVD)

#### Mathematical Foundation
A = UΣV^T where U, V are orthogonal and Σ is diagonal with σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0.

#### Numerical Properties
- **Stability**: SVD algorithms are backward stable
- **Condition Number**: κ₂(A) = σ₁/σᵣ
- **Rank Detection**: rank(A) = number of σᵢ > tol
- **Best Low-Rank Approximation**: Truncated SVD minimizes ||A - Aₖ||₂

#### Implementation Strategy
Our implementation uses power iteration with deflation:
- Efficient for matrices where only a few singular values are needed
- Trade-off between simplicity and optimality

### Cholesky Decomposition

#### Mathematical Foundation
For symmetric positive definite A: A = LL^T or A = U^T U

#### Numerical Considerations
- **Existence**: Requires A to be symmetric positive definite
- **Stability**: Inherently stable (no pivoting needed)
- **Detection**: Failure indicates A is not positive definite
- **Cost**: O(n³/3) operations (half of LU)

#### Pivoting Strategies
For indefinite matrices, use:
- **LDLT**: A = LDL^T with diagonal D
- **Modified Cholesky**: A + E = LL^T with small E

### Eigenvalue Decomposition

#### Mathematical Foundation
Ax = λx where λ are eigenvalues and x are eigenvectors.

#### Algorithms
1. **Power Iteration**: For dominant eigenvalue
2. **QR Algorithm**: For all eigenvalues (most robust)
3. **Divide and Conquer**: For symmetric matrices

#### Numerical Challenges
- **Sensitivity**: Eigenvalues can be very sensitive to perturbations
- **Multiple Eigenvalues**: Special care needed for clusters
- **Complex Eigenvalues**: May require complex arithmetic

#### Our Implementation
Uses power iteration with deflation:
- Simple and robust for well-separated eigenvalues
- May struggle with clustered eigenvalues
- Good for finding a few dominant eigenvalues

## Linear System Solving

### Direct Methods

#### Gaussian Elimination
The foundation of most direct solvers:
1. Forward elimination: Reduce to upper triangular
2. Back substitution: Solve triangular system

#### Pivoting Strategies
- **No Pivoting**: Only for special matrices (e.g., diagonally dominant)
- **Partial Pivoting**: Row exchanges only
- **Complete Pivoting**: Both row and column exchanges (rarely used)

#### Error Analysis
For Ax = b with computed solution x̃:
- **Residual**: r = b - Ax̃
- **Forward Error Bound**: ||x - x̃|| ≤ κ(A) · ||r|| / ||A||

### Iterative Methods

#### Conjugate Gradient (CG)

**Prerequisites**: A must be symmetric positive definite

**Algorithm**: 
1. Choose initial guess x₀
2. Set r₀ = b - Ax₀, p₀ = r₀
3. For k = 0, 1, 2, ...:
   - αₖ = (rₖ^T rₖ) / (pₖ^T Apₖ)
   - xₖ₊₁ = xₖ + αₖpₖ
   - rₖ₊₁ = rₖ - αₖApₖ
   - βₖ = (rₖ₊₁^T rₖ₊₁) / (rₖ^T rₖ)
   - pₖ₊₁ = rₖ₊₁ + βₖpₖ

**Convergence**: 
- Theoretical: Exact solution in n steps
- Practical: Error reduces by factor of (√κ - 1)/(√κ + 1) per iteration

#### Preconditioning
Transform Ax = b to M⁻¹Ax = M⁻¹b where M ≈ A is easy to invert.

**Common Preconditioners**:
- **Diagonal (Jacobi)**: M = diag(A)
- **Incomplete LU**: Approximate LU with fill-in control
- **Multigrid**: Hierarchical approach

### Multigrid Methods

#### Principle
Use hierarchy of grids to accelerate convergence:
- **Smoothing**: Reduce high-frequency error components
- **Restriction**: Transfer to coarser grid
- **Correction**: Solve on coarse grid
- **Interpolation**: Transfer back to fine grid

#### Cycle Types
- **V-cycle**: Down to coarsest, then up
- **W-cycle**: Two recursive calls at each level
- **F-cycle**: Full multigrid with nested iteration

#### Complexity
Optimal multigrid achieves O(n) complexity for n unknowns.

## Special Matrix Structures

### Tridiagonal Matrices
For matrix with diagonals (aᵢ, bᵢ, cᵢ):

**Thomas Algorithm**:
1. Forward sweep: Eliminate subdiagonal
2. Back substitution: O(n) complexity

**Stability**: Stable if matrix is diagonally dominant or symmetric positive definite.

### Toeplitz Matrices
Constant along diagonals: aᵢⱼ = tᵢ₋ⱼ

**Levinson Algorithm**: O(n²) solver using matrix structure
**Applications**: Digital signal processing, time series analysis

### Circulant Matrices
Special case of Toeplitz where elements wrap around.

**FFT-based Solution**: O(n log n) using Fast Fourier Transform
**Eigenvalues**: Given by DFT of first row

## Regularization Techniques

### Tikhonov Regularization
Solve: min ||Ax - b||² + λ||x||²

**Solution**: x = (A^T A + λI)⁻¹ A^T b

**Parameter Selection**:
- **L-curve**: Plot ||Ax - b|| vs ||x||
- **GCV**: Generalized cross-validation
- **Discrepancy Principle**: Choose λ such that ||Ax - b|| ≈ noise level

### Truncated SVD
For A = UΣV^T, use only k largest singular values:
A_k = U_k Σ_k V_k^T

**Solution**: x = V_k Σ_k⁻¹ U_k^T b

**Rank Selection**: Choose k where σₖ₊₁ < tolerance

## Error Analysis and Bounds

### Perturbation Theory

#### Linear Systems
For (A + ΔA)(x + Δx) = b + Δb:
||Δx|| / ||x|| ≤ κ(A) · (||ΔA|| / ||A|| + ||Δb|| / ||b||) / (1 - κ(A)||ΔA||/||A||)

#### Eigenvalue Problems
Eigenvalues are more sensitive than linear systems:
- **Simple eigenvalues**: Perturbation ~ ||ΔA|| / gap
- **Multiple eigenvalues**: Can have O(||ΔA||^(1/m)) perturbation for multiplicity m

### Backward Error Analysis

#### Philosophy
Instead of asking "How close is x̃ to x?", ask "What problem does x̃ solve exactly?"

#### Applications
- **LU**: (A + ΔA)x̃ = b where ||ΔA|| ≤ ρnu||A||
- **QR**: (A + ΔA)x̃ = b where ||ΔA|| ≤ O(u)||A||

Where u is machine epsilon and ρ is growth factor.

## Floating-Point Considerations

### Machine Precision
- **float32**: u ≈ 6 × 10⁻⁸ (about 7 decimal digits)
- **float64**: u ≈ 1 × 10⁻¹⁶ (about 15 decimal digits)

### Catastrophic Cancellation
When subtracting nearly equal numbers, relative error can be amplified.

**Example**: Computing b² - 4ac in quadratic formula
**Solution**: Use alternative formulation or higher precision

### Accumulation of Errors
- **Inner products**: Error grows like √n for n terms
- **Matrix multiplication**: Error bounded by modest factor times machine precision

## Tolerance Selection Guidelines

### General Principles
- **Relative tolerance**: Often more appropriate than absolute
- **Problem-dependent**: Consider physical significance
- **Conservative**: Better to be too strict than too loose

### Recommended Values
```rust
// For float32
const MACHINE_EPS: f32 = 1.2e-7;
const SQRT_EPS: f32 = 3.5e-4;

// Typical tolerances
let solve_tolerance = 100.0 * MACHINE_EPS;     // ~1e-5
let rank_tolerance = SQRT_EPS;                 // ~3e-4
let convergence_tolerance = 1e-6;              // User-specified
```

### Adaptive Tolerances
Consider scaling tolerance based on:
- Matrix condition number
- Problem size
- Required accuracy

## References and Further Reading

### Classical Texts
1. Golub & Van Loan: "Matrix Computations" (4th ed.)
2. Trefethen & Bau: "Numerical Linear Algebra"
3. Higham: "Accuracy and Stability of Numerical Algorithms"

### Modern Developments
1. Demmel: "Applied Numerical Linear Algebra"
2. Stewart: "Matrix Algorithms"
3. Saad: "Iterative Methods for Sparse Linear Systems"

### Software Libraries
1. LAPACK: Reference implementation
2. Intel MKL: Optimized implementation
3. MAGMA: GPU-accelerated linear algebra