//! Matrix property computation functions
//!
//! This module provides functions for computing fundamental matrix properties
//! including rank, condition number, determinant, matrix inverse, and pseudoinverse.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

use super::core::NormOrd;
use super::decompositions::svd;

/// Compute matrix rank using SVD
///
/// ## Mathematical Background
///
/// The rank of a matrix A is the dimension of its column space (or row space).
/// Equivalently, it's the number of linearly independent columns (or rows).
///
/// Using SVD, A = UΣV^T, the rank equals the number of non-zero singular values:
/// ```text
/// rank(A) = |{i : σᵢ > tolerance}|
/// ```text
///
/// ## Numerical Considerations
///
/// Due to floating-point arithmetic, singular values are rarely exactly zero.
/// The tolerance parameter determines the threshold below which singular values
/// are considered "numerically zero."
///
/// ## Parameters
/// * `tensor` - Input matrix A (m×n)
/// * `tol` - Tolerance for determining rank (default: 1e-6)
///
/// ## Returns
/// * Scalar tensor containing the rank of the matrix
///
/// ## Applications
/// - **Linear independence**: Determine if columns/rows are independent
/// - **Solvability**: rank(A) = rank([A|b]) for consistent systems
/// - **Dimensionality**: Effective dimension of data in matrix
/// - **Condition analysis**: Full rank indicates non-singularity
pub fn matrix_rank(tensor: &Tensor, tol: Option<f32>) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Matrix rank requires 2D tensor",
            "matrix_rank",
        ));
    }

    // Placeholder implementation using SVD
    let (_u, s, _vt) = svd(tensor, false)?;
    let s_data = s.data()?.clone();

    let tolerance = tol.unwrap_or(1e-6);
    let rank = s_data.iter().filter(|&&val| val > tolerance).count() as f32;

    Tensor::from_data(vec![rank], vec![], tensor.device())
}

/// Compute matrix condition number
///
/// ## Mathematical Background
///
/// The condition number measures how sensitive a matrix's inverse is to small
/// changes in the matrix. For the 2-norm (spectral norm):
/// ```text
/// κ(A) = ||A||₂ ||A⁻¹||₂ = σₘₐₓ(A) / σₘᵢₙ(A)
/// ```text
///
/// where σₘₐₓ and σₘᵢₙ are the largest and smallest singular values.
///
/// ## Interpretation
///
/// - **κ(A) = 1**: Perfectly conditioned (orthogonal/unitary matrices)
/// - **κ(A) < 100**: Well-conditioned, reliable numerical computations
/// - **κ(A) > 10¹²**: Ill-conditioned, results may be unreliable
/// - **κ(A) = ∞**: Singular matrix, no unique solution
///
/// ## Condition Number Types
///
/// Different norms give different condition numbers:
/// - **2-norm**: κ₂(A) = σₘₐₓ/σₘᵢₙ (most common)
/// - **1-norm**: κ₁(A) = ||A||₁ ||A⁻¹||₁
/// - **∞-norm**: κ∞(A) = ||A||∞ ||A⁻¹||∞
/// - **Frobenius**: κF(A) = ||A||F ||A⁻¹||F
///
/// ## Parameters
/// * `tensor` - Input matrix A (must be square)
/// * `ord` - Norm type for condition number (currently uses 2-norm)
///
/// ## Returns
/// * Scalar tensor containing the condition number
///
/// ## Applications
/// - **Numerical stability**: Assess reliability of linear system solutions
/// - **Error analysis**: Bound propagation of input errors to output
/// - **Algorithm selection**: Choose appropriate solution method
/// - **Regularization**: Determine need for regularization in regression
pub fn cond(tensor: &Tensor, _ord: Option<NormOrd>) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Condition number requires 2D tensor",
            "cond",
        ));
    }

    // For SVD-based condition number (most common)
    let (_u, s, _vt) = svd(tensor, false)?;
    let s_data = s.data()?.clone();

    if s_data.is_empty() {
        return Tensor::from_data(vec![f32::INFINITY], vec![], tensor.device());
    }

    let max_s = s_data.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_s = s_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    let cond_num = if min_s > 1e-12 {
        max_s / min_s
    } else {
        f32::INFINITY
    };

    Tensor::from_data(vec![cond_num], vec![], tensor.device())
}

/// Compute matrix determinant
///
/// ## Mathematical Background
///
/// The determinant is a scalar value that provides important information about
/// a square matrix. For an n×n matrix A:
///
/// ### Geometric Interpretation
/// - **Volume scaling**: |det(A)| = volume scaling factor of linear transformation
/// - **Orientation**: sign(det(A)) indicates orientation preservation
///
/// ### Algebraic Properties
/// 1. **Multiplicativity**: det(AB) = det(A) det(B)
/// 2. **Transpose**: det(A^T) = det(A)
/// 3. **Inverse**: det(A⁻¹) = 1/det(A) (if A is invertible)
/// 4. **Scalar multiple**: det(cA) = cⁿ det(A) for n×n matrix
///
/// ### Computation Methods
/// - **LU decomposition**: det(A) = det(P) ∏ᵢ Uᵢᵢ
/// - **Cofactor expansion**: Recursive expansion along rows/columns
/// - **SVD**: det(A) = ∏ᵢ σᵢ (product of singular values)
///
/// ## Special Cases
/// - **det(A) = 0**: Matrix is singular (non-invertible)
/// - **det(A) > 0**: Preserves orientation
/// - **det(A) < 0**: Reverses orientation
/// - **|det(A)| = 1**: Volume-preserving transformation
///
/// ## Parameters
/// * `tensor` - Square matrix A (n×n)
///
/// ## Returns
/// * Scalar tensor containing det(A)
///
/// ## Applications
/// - **Invertibility**: det(A) ≠ 0 ⟺ A is invertible
/// - **Volume computation**: Volume of parallelepiped from matrix columns
/// - **Characteristic polynomial**: det(A - λI) = 0 for eigenvalues λ
/// - **Cramer's rule**: Solution to Ax = b using determinants
///
/// ## Note
/// This is currently a placeholder returning 1.0. Production implementation
/// should use LU decomposition with partial pivoting for numerical stability.
pub fn det(tensor: &Tensor) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Determinant requires 2D tensor",
            "det",
        ));
    }

    let shape = tensor.shape();
    let dims = shape.dims();
    if dims[0] != dims[1] {
        return Err(TorshError::invalid_argument_with_context(
            "Determinant requires square matrix",
            "det",
        ));
    }

    // Placeholder implementation
    // In a real implementation, we would use LU decomposition
    Tensor::from_data(vec![1.0], vec![], tensor.device())
}

/// Matrix Inverse
///
/// Computes the inverse of a square matrix A such that:
/// ```text
/// A A⁻¹ = A⁻¹ A = I
/// ```text
///
/// ## Mathematical Definition
///
/// For a square matrix A ∈ ℝⁿˣⁿ, the inverse A⁻¹ exists if and only if A is non-singular
/// (i.e., det(A) ≠ 0). The inverse satisfies:
/// - **Identity property**: A A⁻¹ = A⁻¹ A = I
/// - **Uniqueness**: If the inverse exists, it is unique
///
/// ## Mathematical Properties
///
/// 1. **(AB)⁻¹ = B⁻¹ A⁻¹**: Inverse of product (order reverses)
/// 2. **(A⁻¹)⁻¹ = A**: Inverse of inverse
/// 3. **(A^T)⁻¹ = (A⁻¹)^T**: Inverse of transpose
/// 4. **det(A⁻¹) = 1/det(A)**: Determinant of inverse
/// 5. **||A⁻¹||₂ = 1/σₘᵢₙ(A)**: Spectral norm of inverse
///
/// ## Gauss-Jordan Algorithm
///
/// This implementation uses Gauss-Jordan elimination with partial pivoting:
/// ```text
/// [A | I] → [I | A⁻¹]
/// ```text
///
/// 1. **Augment**: Create [A | I] where I is the identity matrix
/// 2. **Forward elimination**: Reduce A to row echelon form
/// 3. **Backward substitution**: Transform to reduced row echelon form [I | A⁻¹]
/// 4. **Extract**: A⁻¹ is the right half of the augmented matrix
///
/// ## Condition Number
///
/// The condition number κ(A) = ||A|| ||A⁻¹|| measures numerical stability:
/// - **κ(A) = 1**: Perfectly conditioned (orthogonal matrices)
/// - **κ(A) >> 1**: Ill-conditioned (near-singular)
/// - **κ(A) = ∞**: Singular (non-invertible)
///
/// ## Alternative Methods
///
/// For specific matrix types, more efficient algorithms exist:
/// - **Positive definite**: A⁻¹ = (L⁻¹)^T L⁻¹ via Cholesky L L^T = A
/// - **Symmetric**: A⁻¹ = V Λ⁻¹ V^T via eigendecomposition A = V Λ V^T
/// - **Well-conditioned**: A⁻¹ = V Σ⁻¹ U^T via SVD A = U Σ V^T
///
/// ## Parameters
/// - `tensor`: Input matrix A (n×n, must be non-singular)
///
/// ## Returns
/// Inverse matrix A⁻¹ such that A A⁻¹ = I
///
/// ## Applications
/// - **Linear systems**: Solve Ax = b via x = A⁻¹b (when multiple RHS)
/// - **Least squares**: Normal equations solution (X^T X)⁻¹ X^T y
/// - **Covariance**: Precision matrix Ω = Σ⁻¹
/// - **Kalman filtering**: Innovation covariance updates
/// - **Statistics**: Fisher information matrix inversion
///
/// ## Numerical Considerations
/// - **Avoid when possible**: Use solve() for Ax = b instead of x = A⁻¹b
/// - **Check condition number**: Large κ(A) indicates numerical instability
/// - **Use pseudoinverse**: For singular/rectangular matrices, use pinv()
pub fn inv(tensor: &Tensor) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Matrix inverse requires 2D tensor",
            "inv",
        ));
    }

    let shape = tensor.shape();
    let dims = shape.dims();
    if dims[0] != dims[1] {
        return Err(TorshError::invalid_argument_with_context(
            "Matrix inverse requires square matrix",
            "inv",
        ));
    }

    // Implement matrix inverse using Gauss-Jordan elimination
    let n = dims[0];
    let data = tensor.data()?;

    // Create augmented matrix [A | I] for Gauss-Jordan elimination
    let mut augmented = vec![0.0f32; n * (2 * n)];

    // Copy original matrix to left side
    for i in 0..n {
        for j in 0..n {
            augmented[i * (2 * n) + j] = data[i * n + j];
        }
    }

    // Add identity matrix to right side
    for i in 0..n {
        augmented[i * (2 * n) + (n + i)] = 1.0;
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if augmented[k * (2 * n) + i].abs() > augmented[max_row * (2 * n) + i].abs() {
                max_row = k;
            }
        }

        // Check for singular matrix
        if augmented[max_row * (2 * n) + i].abs() < 1e-10 {
            return Err(TorshError::invalid_argument_with_context(
                "Matrix is singular or near-singular",
                "inv",
            ));
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = augmented[i * (2 * n) + j];
                augmented[i * (2 * n) + j] = augmented[max_row * (2 * n) + j];
                augmented[max_row * (2 * n) + j] = temp;
            }
        }

        // Scale pivot row
        let pivot = augmented[i * (2 * n) + i];
        for j in 0..(2 * n) {
            augmented[i * (2 * n) + j] /= pivot;
        }

        // Eliminate other rows
        for k in 0..n {
            if k != i {
                let factor = augmented[k * (2 * n) + i];
                for j in 0..(2 * n) {
                    augmented[k * (2 * n) + j] -= factor * augmented[i * (2 * n) + j];
                }
            }
        }
    }

    // Extract inverse matrix from right side of augmented matrix
    let mut inv_data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv_data[i * n + j] = augmented[i * (2 * n) + (n + j)];
        }
    }

    Tensor::from_data(inv_data, dims.to_vec(), tensor.device())
}

/// Pseudo-inverse (Moore-Penrose inverse)
///
/// ## Mathematical Background
///
/// The Moore-Penrose pseudoinverse A⁺ generalizes matrix inversion to
/// non-square and singular matrices. For any m×n matrix A, A⁺ is the
/// unique n×m matrix satisfying:
///
/// 1. **A A⁺ A = A**: A⁺ is a generalized inverse
/// 2. **A⁺ A A⁺ = A⁺**: A⁺ is reflexive
/// 3. **(A A⁺)^T = A A⁺**: A A⁺ is symmetric
/// 4. **(A⁺ A)^T = A⁺ A**: A⁺ A is symmetric
///
/// ## SVD Construction
///
/// Given the SVD A = U Σ V^T, the pseudoinverse is:
/// ```text
/// A⁺ = V Σ⁺ U^T
/// ```text
/// where Σ⁺ is formed by:
/// - Transposing Σ
/// - Taking reciprocal of non-zero singular values
/// - Setting zero/small singular values to zero
///
/// ## Properties
///
/// 1. **Unique**: The pseudoinverse is uniquely defined
/// 2. **Inverse relationship**: If A is invertible, then A⁺ = A⁻¹
/// 3. **Least squares**: x = A⁺b minimizes ||Ax - b||₂
/// 4. **Best approximation**: A⁺ gives minimum norm solution when multiple solutions exist
///
/// ## Rank and Dimensions
///
/// For an m×n matrix A with rank r:
/// - **Full column rank** (r = n): A⁺ = (A^T A)⁻¹ A^T (left inverse)
/// - **Full row rank** (r = m): A⁺ = A^T (A A^T)⁻¹ (right inverse)
/// - **Rank deficient** (r < min(m,n)): Use SVD construction
///
/// ## Parameters
/// * `tensor` - Input matrix A (m×n, can be rectangular or singular)
/// * `rcond` - Relative condition number for singular value cutoff (default: 1e-15)
///
/// ## Returns
/// * Pseudoinverse A⁺ (n×m matrix)
///
/// ## Applications
/// - **Least squares**: Overdetermined systems Ax ≈ b
/// - **Underdetermined systems**: Minimum norm solutions to Ax = b
/// - **Data analysis**: Principal component regression
/// - **Signal processing**: Inverse filtering and deconvolution
/// - **Control theory**: System inversion and feedforward design
///
/// ## Numerical Considerations
/// - **Singular value threshold**: Small singular values indicate ill-conditioning
/// - **Condition number**: Large condition numbers lead to amplified noise
/// - **Regularization**: Consider Tikhonov regularization for ill-posed problems
///
/// ## Note
/// This is currently a placeholder implementation returning the transpose.
/// Production code should use the SVD-based construction described above.
pub fn pinv(tensor: &Tensor, rcond: Option<f32>) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Pseudo-inverse requires 2D tensor",
            "pinv",
        ));
    }

    let _dims = tensor.shape().dims();
    let _rcond = rcond.unwrap_or(1e-15);

    // Placeholder implementation using SVD
    // A+ = V * S+ * U^T where S+ is the pseudo-inverse of the singular values
    let (_u, _s, vt) = svd(tensor, false)?;

    // For now, return transpose as a simple approximation
    Ok(vt.transpose(-2, -1)?)
}
