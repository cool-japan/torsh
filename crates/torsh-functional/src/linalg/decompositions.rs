//! Matrix decomposition operations
//!
//! This module provides fundamental matrix decompositions including LU, QR, Cholesky,
//! SVD, and eigenvalue decomposition. These decompositions are building blocks for
//! many numerical algorithms and have extensive applications in machine learning,
//! scientific computing, and engineering.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::{
    creation::{eye, ones},
    Tensor,
};

/// LU decomposition with partial pivoting
///
/// Computes the LU decomposition of a square matrix A:
/// ```text
/// P A = L U
/// ```text
///
/// ## Mathematical Background
///
/// LU decomposition factors a matrix into:
/// - **P**: Permutation matrix (partial pivoting for numerical stability)
/// - **L**: Lower triangular matrix with unit diagonal
/// - **U**: Upper triangular matrix
///
/// ## Applications
/// - **Linear systems**: Solve Ax = b via forward/backward substitution
/// - **Matrix inversion**: A⁻¹ = U⁻¹ L⁻¹ P^T
/// - **Determinant**: det(A) = det(P) × ∏ᵢ Uᵢᵢ
///
/// ## Parameters
/// * `tensor` - Square matrix to decompose
///
/// ## Returns
/// * Tuple (P, L, U) where P A = L U
///
/// ## Note
/// This is currently a placeholder implementation. A production version would
/// implement the actual LU decomposition algorithm with partial pivoting.
pub fn lu(tensor: &Tensor) -> TorshResult<(Tensor, Tensor, Tensor)> {
    // Validate input
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "LU decomposition requires 2D tensor",
            "lu",
        ));
    }

    let n = tensor.shape().dims()[0];
    let m = tensor.shape().dims()[1];

    if n != m {
        return Err(TorshError::invalid_argument_with_context(
            "LU decomposition requires square matrix",
            "lu",
        ));
    }

    // This is a placeholder implementation
    // Real implementation would perform actual LU decomposition
    let p = eye(n)?;
    let l = eye(n)?;
    let u = tensor.clone();

    Ok((p, l, u))
}

/// QR Decomposition
///
/// Computes the QR decomposition of a matrix A:
/// ```text
/// A = QR
/// ```text
///
/// ## Mathematical Definition
///
/// For an m×n matrix A, the QR decomposition is a factorization:
/// - **Q**: m×m orthogonal matrix (when reduced=false) or m×min(m,n) (when reduced=true)
/// - **R**: m×n upper triangular matrix (when reduced=false) or min(m,n)×n (when reduced=true)
///
/// ## Mathematical Properties
///
/// 1. **Orthogonality**: Q^T Q = I (Q has orthonormal columns)
/// 2. **Upper triangular**: R[i,j] = 0 for i > j
/// 3. **Uniqueness**: If A has full column rank and R has positive diagonal elements,
///    then the QR decomposition is unique
/// 4. **Determinant**: det(A) = det(R) (since det(Q) = ±1)
///
/// ## Gram-Schmidt Process
///
/// The QR decomposition can be computed via Gram-Schmidt orthogonalization:
/// ```text
/// q₁ = a₁ / ||a₁||
/// q₂ = (a₂ - (q₁·a₂)q₁) / ||(a₂ - (q₁·a₂)q₁)||
/// qₖ = (aₖ - Σᵢ₌₁ᵏ⁻¹(qᵢ·aₖ)qᵢ) / ||aₖ - Σᵢ₌₁ᵏ⁻¹(qᵢ·aₖ)qᵢ||
/// ```text
///
/// ## Parameters
/// - `tensor`: Input matrix A (m×n)
/// - `reduced`: If true, returns "thin" QR with Q (m×min(m,n)) and R (min(m,n)×n).
///             If false, returns "full" QR with Q (m×m) and R (m×n).
///
/// ## Returns
/// Tuple (Q, R) where:
/// - Q: Orthogonal matrix with orthonormal columns
/// - R: Upper triangular matrix
///
/// ## Applications
/// - **Least squares**: Solve Ax = b via Rx = Q^T b
/// - **Eigenvalue algorithms**: QR iteration for eigenvalue computation
/// - **Orthogonalization**: Extract orthonormal basis from column space
/// - **Matrix inversion**: A⁻¹ = R⁻¹ Q^T (when A is square and invertible)
pub fn qr(tensor: &Tensor, reduced: bool) -> TorshResult<(Tensor, Tensor)> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "QR decomposition requires 2D tensor",
            "qr",
        ));
    }

    // Use torsh-linalg's QR implementation
    // Note: torsh-linalg::decomposition::qr always returns reduced form
    // We need to adapt it if full_matrices is requested
    let (q, r) = torsh_linalg::decomposition::qr(tensor)?;

    if reduced {
        // Already in reduced form
        Ok((q, r))
    } else {
        // Need to expand Q to full m×m matrix
        let shape = tensor.shape();
        let dims = shape.dims();
        let m = dims[0];
        let n = dims[1];
        let k = m.min(n);

        if k == m {
            // Already full size
            Ok((q, r))
        } else {
            // Expand Q from m×k to m×m by padding with identity
            let mut q_data = vec![0.0f32; m * m];

            // Copy existing Q
            let q_vec = q.to_vec()?;
            for i in 0..m {
                for j in 0..k {
                    q_data[i * m + j] = q_vec[i * k + j];
                }
            }

            // Add identity for remaining columns
            for i in k..m {
                q_data[i * m + i] = 1.0;
            }

            let q_full = Tensor::from_data(q_data, vec![m, m], tensor.device())?;
            Ok((q_full, r))
        }
    }
}

/// Cholesky Decomposition
///
/// Computes the Cholesky decomposition of a positive definite matrix A:
/// ```text
/// A = L L^T  (lower form)
/// A = U^T U  (upper form)
/// ```text
///
/// ## Mathematical Definition
///
/// For a symmetric positive definite matrix A, the Cholesky decomposition uniquely factors A as:
/// - **Lower form**: A = L L^T where L is lower triangular with positive diagonal
/// - **Upper form**: A = U^T U where U is upper triangular with positive diagonal
///
/// ## Mathematical Properties
///
/// 1. **Uniqueness**: The Cholesky factor is unique for positive definite matrices
/// 2. **Efficiency**: Requires ~n³/3 operations (half of LU decomposition)
/// 3. **Numerical stability**: More stable than LU for positive definite systems
/// 4. **Determinant**: det(A) = (∏ᵢ Lᵢᵢ)² = ∏ᵢ Lᵢᵢ²
///
/// ## Cholesky Algorithm
///
/// For lower triangular L where A = L L^T:
/// ```text
/// for i = 0 to n-1:
///     Lᵢᵢ = √(Aᵢᵢ - Σⱼ₌₀ⁱ⁻¹ Lᵢⱼ²)
///     for k = i+1 to n-1:
///         Lₖᵢ = (Aₖᵢ - Σⱼ₌₀ⁱ⁻¹ Lₖⱼ Lᵢⱼ) / Lᵢᵢ
/// ```text
///
/// ## Parameters
/// - `tensor`: Input matrix A (n×n, must be symmetric positive definite)
/// - `upper`: If true, returns upper triangular U such that A = U^T U.
///           If false, returns lower triangular L such that A = L L^T.
///
/// ## Returns
/// Cholesky factor:
/// - Lower triangular L (if upper=false)
/// - Upper triangular U (if upper=true)
///
/// ## Applications
/// - **Linear systems**: Solve Ax = b via Ly = b, L^T x = y
/// - **Matrix inversion**: A⁻¹ = (L^T)⁻¹ L⁻¹
/// - **Determinant**: det(A) = ∏ᵢ Lᵢᵢ²
/// - **Gaussian sampling**: Generate x ~ N(μ, A) via x = μ + L z where z ~ N(0,I)
/// - **Optimization**: Newton's method with positive definite Hessians
pub fn cholesky(tensor: &Tensor, upper: bool) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Cholesky decomposition requires 2D tensor",
            "cholesky",
        ));
    }

    let binding = tensor.shape();
    let dims = binding.dims();
    if dims[0] != dims[1] {
        return Err(TorshError::invalid_argument_with_context(
            "Cholesky decomposition requires square matrix",
            "cholesky",
        ));
    }

    // Use torsh-linalg's Cholesky implementation
    torsh_linalg::decomposition::cholesky(tensor, upper)
}

/// Singular Value Decomposition (SVD)
///
/// Computes the singular value decomposition of a matrix A:
/// ```text
/// A = U Σ V^T
/// ```text
///
/// ## Mathematical Definition
///
/// For an m×n matrix A, the SVD is a factorization:
/// - **U**: m×m orthogonal matrix (left singular vectors)
/// - **Σ**: m×n diagonal matrix (singular values σ₁ ≥ σ₂ ≥ ... ≥ σₘᵢₙ ≥ 0)
/// - **V^T**: n×n orthogonal matrix (right singular vectors, transposed)
///
/// ## Mathematical Properties
///
/// 1. **Orthogonality**: U^T U = I, V^T V = I
/// 2. **Rank**: rank(A) = number of non-zero singular values
/// 3. **Norm preservation**: ||A||₂ = σ₁ (largest singular value)
/// 4. **Eckart-Young theorem**: Best rank-k approximation A_k = U_k Σ_k V_k^T
///
/// ## Parameters
/// - `tensor`: Input matrix A (m×n)
/// - `full_matrices`: If true, return full U (m×m) and V^T (n×n).
///                   If false, return reduced form U (m×min(m,n)) and V^T (min(m,n)×n)
///
/// ## Returns
/// Tuple (U, Σ, V^T) where:
/// - U: Left singular vectors
/// - Σ: Singular values (1D tensor)
/// - V^T: Right singular vectors (transposed)
///
/// ## Example Applications
/// - **Matrix pseudoinverse**: A⁺ = V Σ⁺ U^T
/// - **Principal Component Analysis**: Components from U or V
/// - **Low-rank approximation**: A ≈ Σᵢ₌₁ᵏ σᵢ uᵢ vᵢ^T
/// - **Condition number**: κ(A) = σ₁/σₘᵢₙ
pub fn svd(tensor: &Tensor, full_matrices: bool) -> TorshResult<(Tensor, Tensor, Tensor)> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "SVD requires 2D tensor",
            "svd",
        ));
    }

    // Use torsh-linalg's SVD implementation
    torsh_linalg::decomposition::svd(tensor, full_matrices)
}

/// Eigenvalue decomposition for square matrices
///
/// Computes the eigenvalue decomposition of a square matrix A:
/// ```text
/// A V = V Λ  or  A = V Λ V⁻¹
/// ```text
///
/// ## Mathematical Background
///
/// For a square matrix A ∈ ℝⁿˣⁿ, the eigendecomposition finds:
/// - **Eigenvalues** λᵢ: Scalars such that A vᵢ = λᵢ vᵢ
/// - **Eigenvectors** vᵢ: Non-zero vectors satisfying the eigenvalue equation
///
/// ## Properties
/// - **Characteristic polynomial**: det(A - λI) = 0
/// - **Trace**: tr(A) = Σᵢ λᵢ
/// - **Determinant**: det(A) = ∏ᵢ λᵢ
/// - **Spectral radius**: ρ(A) = max |λᵢ|
///
/// ## Parameters
/// * `tensor` - Square matrix to decompose
///
/// ## Returns
/// * Tuple (eigenvalues, eigenvectors) where eigenvalues are 1D and eigenvectors are 2D
///
/// ## Applications
/// - **Principal Component Analysis**: Covariance matrix eigendecomposition
/// - **Stability analysis**: Eigenvalues determine system stability
/// - **Matrix powers**: A^k = V Λ^k V⁻¹
/// - **Matrix functions**: f(A) = V f(Λ) V⁻¹
pub fn eig(tensor: &Tensor) -> TorshResult<(Tensor, Tensor)> {
    use scirs2_core::ndarray::Array2;

    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Eigenvalue decomposition requires 2D tensor",
            "eig",
        ));
    }

    let shape = tensor.shape();
    let dims = shape.dims();
    if dims[0] != dims[1] {
        return Err(TorshError::invalid_argument_with_context(
            "Eigenvalue decomposition requires square matrix",
            "eig",
        ));
    }

    let n = dims[0];
    let device = tensor.device();

    // Work in f64 internally for numerical accuracy, then narrow back to f32.
    // `to_vec` yields the matrix in row-major order, so element (i, j) lives at
    // index `i * n + j`.
    let data: Vec<f64> = tensor.to_vec()?.into_iter().map(|x| x as f64).collect();

    // The matrix that is actually decomposed (the symmetric part for the
    // symmetric branch, the original matrix otherwise). Kept for the residual
    // verification below.
    let decomposed: Vec<f64>;

    // Eigenpairs as (eigenvalue, unit-norm eigenvector) in f64.
    let mut pairs: Vec<(f64, Vec<f64>)> = if matrix_is_symmetric(&data, n) {
        // Symmetric / Hermitian case. scirs2-linalg's symmetric eigensolver is
        // numerically robust for exactly the degenerate cases that break naive
        // power iteration: repeated eigenvalues (e.g. diag(2, 2, 5)) and
        // rank-deficient matrices with zero eigenvalues both yield a full,
        // orthonormal eigenvector basis. We symmetrize first so the (A + Aᵀ)/2
        // handed to the solver is exactly symmetric in f64.
        let symmetric = symmetrize(&data, n);
        let a = Array2::from_shape_vec((n, n), symmetric.clone()).map_err(|e| {
            TorshError::ComputeError(format!("eig: matrix construction failed: {e}"))
        })?;
        decomposed = symmetric;
        let (values, vectors) = scirs2_linalg::eigh(&a.view(), None).map_err(|e| {
            TorshError::ComputeError(format!("eig: symmetric eigensolver failed: {e}"))
        })?;
        if values.len() != n {
            return Err(TorshError::ComputeError(format!(
                "eig: solver returned {} eigenvalues for an {n}x{n} matrix",
                values.len()
            )));
        }
        (0..n)
            .map(|j| {
                let column = normalized((0..n).map(|i| vectors[[i, j]]).collect());
                (values[j], column)
            })
            .collect()
    } else {
        // General (non-symmetric) case. The QR-based solver returns complex
        // eigenpairs; a real eigendecomposition only exists when every
        // eigenvalue is real. The vectors it accumulates are Schur vectors,
        // which coincide with eigenvectors only for normal matrices, so each
        // returned pair is verified below and an honest error is produced
        // otherwise — never a fabricated decomposition.
        let a = Array2::from_shape_vec((n, n), data.clone()).map_err(|e| {
            TorshError::ComputeError(format!("eig: matrix construction failed: {e}"))
        })?;
        decomposed = data.clone();
        let (values, vectors) = scirs2_linalg::eig(&a.view(), None)
            .map_err(|e| TorshError::ComputeError(format!("eig: eigensolver failed: {e}")))?;
        if values.len() != n {
            return Err(TorshError::ComputeError(format!(
                "eig: solver returned {} eigenvalues for an {n}x{n} matrix",
                values.len()
            )));
        }

        let scale = data.iter().fold(1.0f64, |m, &x| m.max(x.abs()));
        for k in 0..n {
            if values[k].im.abs() > 1e-6 * scale {
                return Err(TorshError::ComputeError(format!(
                    "eig: matrix has a complex eigenvalue ({:.6}{:+.6}i); a real \
                     eigendecomposition does not exist for this matrix",
                    values[k].re, values[k].im
                )));
            }
        }

        (0..n)
            .map(|j| {
                // For a real eigenvalue the eigenvector spans a real line, so the
                // complex column is a complex multiple of a real vector and its
                // real and imaginary parts are parallel. Keep whichever part is
                // larger to avoid selecting a (near) zero vector.
                let re: Vec<f64> = (0..n).map(|i| vectors[[i, j]].re).collect();
                let im: Vec<f64> = (0..n).map(|i| vectors[[i, j]].im).collect();
                let re_norm = re.iter().map(|x| x * x).sum::<f64>().sqrt();
                let im_norm = im.iter().map(|x| x * x).sum::<f64>().sqrt();
                let column = normalized(if re_norm >= im_norm { re } else { im });
                (values[j].re, column)
            })
            .collect()
    };

    // Order eigenpairs by descending eigenvalue (dominant first): a stable,
    // PCA-friendly convention that keeps the dominant eigenpair in column 0.
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Anti-fabrication guard: verify A v ≈ λ v for every returned eigenpair.
    // A defective / non-diagonalizable matrix (or a solver failure) surfaces as
    // an honest error here instead of a plausible-but-wrong decomposition.
    let residual = max_relative_residual(&decomposed, n, &pairs);
    if residual > 1e-3 {
        return Err(TorshError::ComputeError(format!(
            "eig: failed to compute an accurate eigendecomposition (max relative \
             residual {residual:.3e}); the matrix is likely defective / \
             non-diagonalizable"
        )));
    }

    // Pack results: eigenvalues as a length-n vector and eigenvectors as the
    // columns of an n x n matrix (column j is the eigenvector for eigenvalue j).
    let mut eigenvalue_data = Vec::with_capacity(n);
    let mut eigenvector_data = vec![0.0f32; n * n];
    for (col, (lambda, vector)) in pairs.iter().enumerate() {
        eigenvalue_data.push(*lambda as f32);
        for (row, &value) in vector.iter().enumerate() {
            eigenvector_data[row * n + col] = value as f32;
        }
    }

    let eigenvalues = Tensor::from_data(eigenvalue_data, vec![n], device)?;
    let eigenvectors = Tensor::from_data(eigenvector_data, vec![n, n], device)?;
    Ok((eigenvalues, eigenvectors))
}

/// Returns `true` if the row-major `n` x `n` matrix is symmetric within a small
/// relative tolerance.
fn matrix_is_symmetric(data: &[f64], n: usize) -> bool {
    for i in 0..n {
        for j in (i + 1)..n {
            let upper = data[i * n + j];
            let lower = data[j * n + i];
            let scale = upper.abs().max(lower.abs()).max(1.0);
            if (upper - lower).abs() > 1e-6 * scale {
                return false;
            }
        }
    }
    true
}

/// Returns the symmetric part `(A + Aᵀ) / 2` of a row-major `n` x `n` matrix.
fn symmetrize(data: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = 0.5 * (data[i * n + j] + data[j * n + i]);
        }
    }
    out
}

/// Scales a vector to unit Euclidean norm (a numerically zero vector is left
/// untouched).
fn normalized(mut v: Vec<f64>) -> Vec<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-300 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

/// Largest relative residual `‖A v − λ v‖₂ / (|λ| + 1)` over all eigenpairs.
/// `pairs` holds `(λ, v)` columns and `matrix` is the row-major `n` x `n` matrix
/// that was decomposed.
fn max_relative_residual(matrix: &[f64], n: usize, pairs: &[(f64, Vec<f64>)]) -> f64 {
    let mut worst = 0.0f64;
    for pair in pairs {
        let lambda = pair.0;
        let v = &pair.1;
        let mut residual_sq = 0.0f64;
        for i in 0..n {
            let mut av_i = 0.0f64;
            for j in 0..n {
                av_i += matrix[i * n + j] * v[j];
            }
            let diff = av_i - lambda * v[i];
            residual_sq += diff * diff;
        }
        let relative = residual_sq.sqrt() / (lambda.abs() + 1.0);
        if relative > worst {
            worst = relative;
        }
    }
    worst
}

/// Low-rank SVD approximation using randomized algorithms
///
/// Computes an approximate SVD for large matrices using randomized algorithms,
/// which can be much faster than full SVD for low-rank approximations.
///
/// ## Mathematical Background
///
/// Randomized SVD uses random projections to efficiently compute low-rank
/// approximations of large matrices. The algorithm works by:
/// 1. Finding a good subspace Q that captures the range of A
/// 2. Computing B = Q^T A (smaller projected matrix)
/// 3. Computing SVD of B and mapping back to original space
///
/// ## Parameters
/// * `tensor` - Input matrix to decompose
/// * `rank` - Target rank for approximation (default: min(m,n,10))
/// * `niter` - Number of power iterations for accuracy (default: 2)
///
/// ## Returns
/// * Tuple (U, S, V) representing the low-rank SVD approximation
pub fn svd_lowrank(
    tensor: &Tensor,
    rank: Option<usize>,
    niter: Option<usize>,
) -> TorshResult<(Tensor, Tensor, Tensor)> {
    // Validate input
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "SVD requires 2D tensor",
            "svd_lowrank",
        ));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    let k = rank.unwrap_or(m.min(n).min(10));
    let _niter = niter.unwrap_or(2);

    // This is a placeholder implementation
    // Real implementation would use randomized SVD algorithm
    use torsh_tensor::creation::randn;
    let u = randn(&[m, k])?;
    let s = ones(&[k])?;
    let v = randn(&[k, n])?;

    Ok((u, s, v))
}

/// Low-rank PCA approximation
///
/// Computes principal components using randomized SVD for efficient
/// dimensionality reduction of large datasets.
///
/// ## Mathematical Background
///
/// PCA finds the directions of maximum variance in data:
/// 1. Center the data: X̃ = X - μ
/// 2. Compute covariance: C = X̃^T X̃ / (n-1)
/// 3. Find eigenvectors: principal components are eigenvectors of C
///
/// ## Parameters
/// * `tensor` - Input data matrix (samples × features)
/// * `rank` - Number of principal components (default: min dimensions)
/// * `center` - Whether to center the data (default: true)
///
/// ## Returns
/// * Tuple (U, S, V^T) where V^T contains the principal components
pub fn pca_lowrank(
    tensor: &Tensor,
    rank: Option<usize>,
    center: bool,
) -> TorshResult<(Tensor, Tensor, Tensor)> {
    // Validate input
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "PCA requires 2D tensor",
            "pca_lowrank",
        ));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    let k = rank.unwrap_or(m.min(n));

    // For now, simple placeholder implementation
    let mut data_tensor = tensor.clone();
    if center {
        // Center the data (subtract mean)
        // This is a simplified version - real implementation would compute actual mean
        data_tensor = tensor.clone();
    }

    // Use SVD for PCA
    let (u, s, v) = svd_lowrank(&data_tensor, Some(k), None)?;

    // For PCA, we typically return V^T as the principal components
    Ok((u, s, v.transpose(-2, -1)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::eye;

    #[test]
    fn test_eig_identity_returns_ones() {
        // The identity matrix has every eigenvalue equal to 1.
        let identity = eye::<f32>(4).unwrap();
        let (eigenvalues, eigenvectors) = eig(&identity).unwrap();

        assert_eq!(eigenvalues.shape().dims(), &[4]);
        assert_eq!(eigenvectors.shape().dims(), &[4, 4]);

        for &lambda in eigenvalues.to_vec().unwrap().iter() {
            assert!(
                (lambda - 1.0).abs() < 1e-5,
                "identity eigenvalue should be 1.0, got {lambda}"
            );
        }
    }

    #[test]
    fn test_eig_diagonal_exact() {
        // Diagonal matrices have an exact eigendecomposition: the eigenvalues
        // are precisely the diagonal entries.
        let diag = Tensor::from_data(
            vec![3.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 5.0],
            vec![3, 3],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let (eigenvalues, _vectors) = eig(&diag).unwrap();
        let mut values = eigenvalues.to_vec().unwrap();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let expected = [-2.0_f32, 3.0, 5.0];
        for (got, want) in values.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-5,
                "diagonal eigenvalue mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_eig_non_square_errors() {
        let rect = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();
        assert!(eig(&rect).is_err());
    }

    #[test]
    fn test_eig_dominant_eigenpair_residual() {
        // For a symmetric 2x2 matrix [[2, 1], [1, 2]] the eigenvalues are 1 and
        // 3. Verify the dominant eigenpair satisfies A v = lambda v.
        let a = Tensor::from_data(
            vec![2.0, 1.0, 1.0, 2.0],
            vec![2, 2],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let (eigenvalues, eigenvectors) = eig(&a).unwrap();
        let lambda = eigenvalues.to_vec().unwrap()[0];
        let vecs = eigenvectors.to_vec().unwrap();
        // First eigenvector is column 0 of the (2x2) matrix stored row-major.
        let v = [vecs[0], vecs[2]];

        // Compute A v.
        let av0 = 2.0 * v[0] + 1.0 * v[1];
        let av1 = 1.0 * v[0] + 2.0 * v[1];

        assert!(
            (av0 - lambda * v[0]).abs() < 1e-3 && (av1 - lambda * v[1]).abs() < 1e-3,
            "A v should equal lambda v (lambda={lambda}, v=[{}, {}])",
            v[0],
            v[1]
        );
        // The dominant eigenvalue of this matrix is 3.
        assert!(
            (lambda - 3.0).abs() < 1e-3,
            "dominant eigenvalue should be 3.0, got {lambda}"
        );
    }

    /// Asserts that every column eigenvector `v_j` of the returned decomposition
    /// satisfies `A v_j ≈ λ_j v_j` and is non-degenerate. This is the assertion
    /// that fails for a fabricated decomposition.
    fn assert_eigenpairs_valid(
        a: &Tensor,
        eigenvalues: &Tensor,
        eigenvectors: &Tensor,
        n: usize,
        tol: f32,
    ) {
        let a_data = a.to_vec().unwrap();
        let vals = eigenvalues.to_vec().unwrap();
        let vecs = eigenvectors.to_vec().unwrap();

        for col in 0..n {
            let lambda = vals[col];
            let v: Vec<f32> = (0..n).map(|row| vecs[row * n + col]).collect();

            // A genuine eigenvector must be non-trivial (these are unit-norm).
            let v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                v_norm > 0.5,
                "eigenvector {col} is degenerate (norm {v_norm})"
            );

            for row in 0..n {
                let mut av = 0.0f32;
                for k in 0..n {
                    av += a_data[row * n + k] * v[k];
                }
                assert!(
                    (av - lambda * v[row]).abs() < tol,
                    "A v != lambda v at row {row}, col {col}: Av={av}, lambda*v={}",
                    lambda * v[row]
                );
            }
        }
    }

    #[test]
    fn test_eig_repeated_eigenvalue() {
        // A = [[3,1,1],[1,3,1],[1,1,3]] = 2I + J (J all-ones) has spectrum
        // {5, 2, 2}: the eigenvalue 2 has algebraic multiplicity 2. A repeated
        // eigenvalue breaks single-vector power iteration but must be handled by
        // a robust symmetric eigensolver, which returns a full orthonormal basis
        // for the 2-eigenspace.
        let a = Tensor::from_data(
            vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0],
            vec![3, 3],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let (eigenvalues, eigenvectors) = eig(&a).unwrap();
        assert_eq!(eigenvalues.shape().dims(), &[3]);
        assert_eq!(eigenvectors.shape().dims(), &[3, 3]);

        let mut values = eigenvalues.to_vec().unwrap();
        values.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let expected = [2.0_f32, 2.0, 5.0];
        for (got, want) in values.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-3,
                "repeated-eigenvalue spectrum mismatch: got {got}, want {want}"
            );
        }

        // Every returned eigenpair (including both copies of 2) must satisfy
        // A v = lambda v.
        assert_eigenpairs_valid(&a, &eigenvalues, &eigenvectors, 3, 1e-3);
    }

    #[test]
    fn test_eig_rank_deficient_zero_eigenvalue() {
        // The all-ones 3x3 matrix has rank 1 and spectrum {3, 0, 0}: two zero
        // eigenvalues. Naive deflation bails out at the first zero eigenvalue and
        // pads the result with standard basis vectors that are NOT eigenvectors,
        // so `A v = 0` fails for those columns. A correct decomposition returns
        // genuine null-space vectors for the zero eigenvalues.
        let a = Tensor::from_data(
            vec![1.0; 9],
            vec![3, 3],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let (eigenvalues, eigenvectors) = eig(&a).unwrap();

        let mut values = eigenvalues.to_vec().unwrap();
        values.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let expected = [0.0_f32, 0.0, 3.0];
        for (got, want) in values.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-3,
                "rank-deficient spectrum mismatch: got {got}, want {want}"
            );
        }

        // The eigenvectors for the zero eigenvalues must really lie in the null
        // space (A v = 0), which is precisely what the old power-iteration path
        // got wrong.
        assert_eigenpairs_valid(&a, &eigenvalues, &eigenvectors, 3, 1e-3);
    }
}
