//! Advanced matrix operations
//!
//! This module provides advanced matrix operations including:
//! - Hadamard (element-wise) product
//! - Matrix vectorization and reconstruction
//! - Commutator and anti-commutator operations

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

use crate::validate_square_matrix;

/// Compute the Hadamard (element-wise) product of two matrices
///
/// The Hadamard product A ∘ B is the element-wise product of matrices
/// of the same dimensions: (A ∘ B)_{ij} = A_{ij} * B_{ij}
///
/// # Arguments
///
/// * `a` - First matrix (m×n)
/// * `b` - Second matrix (m×n) - must have same dimensions as A
///
/// # Returns
///
/// Element-wise product matrix of same dimensions
///
/// # Properties
///
/// - Commutative: A ∘ B = B ∘ A
/// - Associative: (A ∘ B) ∘ C = A ∘ (B ∘ C)
/// - Distributive over addition: A ∘ (B + C) = (A ∘ B) + (A ∘ C)
/// - (A ∘ B)^T = A^T ∘ B^T
///
/// # Examples
///
/// ```ignore
/// use torsh_linalg::hadamard;
/// let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)?;
/// let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], DeviceType::Cpu)?;
/// let h = hadamard(&a, &b)?;  // [[5, 12], [21, 32]]
/// ```
pub fn hadamard(a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
    if a.shape().dims() != b.shape().dims() {
        return Err(TorshError::InvalidArgument(format!(
            "Hadamard product requires tensors with same dimensions, got {:?} and {:?}",
            a.shape().dims(),
            b.shape().dims()
        )));
    }

    // Element-wise multiplication
    a.mul(b)
}

/// Vectorize a matrix by stacking its columns into a single vector
///
/// The vec operation converts a matrix A (m×n) into a column vector of length mn
/// by stacking columns: vec(A) = [a₁₁, a₂₁, ..., aₘ₁, a₁₂, a₂₂, ..., aₘₙ]^T
///
/// # Arguments
///
/// * `tensor` - Matrix to vectorize (m×n)
///
/// # Returns
///
/// Column vector of length mn
///
/// # Properties
///
/// - vec(A + B) = vec(A) + vec(B)
/// - vec(cA) = c·vec(A) for scalar c
/// - Useful for matrix equations: vec(AXB) = (B^T ⊗ A)vec(X)
///
/// # Examples
///
/// ```ignore
/// use torsh_linalg::vec_matrix;
/// let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)?;
/// let v = vec_matrix(&a)?;  // [1, 2, 3, 4] as column vector
/// ```
pub fn vec_matrix(tensor: &Tensor) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(format!(
            "vec operation requires 2D tensor, got {}D",
            tensor.shape().ndim()
        )));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    let total_len = m * n;
    let mut vec_data = Vec::with_capacity(total_len);

    // Stack columns: iterate column-wise (column-major order)
    for j in 0..n {
        for i in 0..m {
            vec_data.push(tensor.get(&[i, j])?);
        }
    }

    Tensor::from_data(vec_data, vec![total_len], tensor.device())
}

/// Inverse of vec operation: reshape a vector into a matrix
///
/// The unvec operation converts a vector of length mn back into a matrix (m×n)
/// by unstacking it column-wise.
///
/// # Arguments
///
/// * `tensor` - Vector to reshape (length mn)
/// * `rows` - Number of rows in output matrix
/// * `cols` - Number of columns in output matrix
///
/// # Returns
///
/// Matrix of dimensions (rows×cols)
///
/// # Properties
///
/// - unvec(vec(A)) = A
/// - unvec is the left inverse of vec
///
/// # Examples
///
/// ```ignore
/// use torsh_linalg::unvec_matrix;
/// let v = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu)?;
/// let a = unvec_matrix(&v, 2, 2)?;  // [[1, 3], [2, 4]]
/// ```
pub fn unvec_matrix(tensor: &Tensor, rows: usize, cols: usize) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(format!(
            "unvec operation requires 1D tensor, got {}D",
            tensor.shape().ndim()
        )));
    }

    let vec_len = tensor.shape().dims()[0];
    if vec_len != rows * cols {
        return Err(TorshError::InvalidArgument(format!(
            "Vector length {} does not match matrix dimensions {}x{} = {}",
            vec_len,
            rows,
            cols,
            rows * cols
        )));
    }

    let mut matrix_data = vec![0.0f32; rows * cols];

    // Unstack columns: fill column-wise (column-major order)
    for j in 0..cols {
        for i in 0..rows {
            let vec_idx = j * rows + i;
            matrix_data[i * cols + j] = tensor.get(&[vec_idx])?;
        }
    }

    Tensor::from_data(matrix_data, vec![rows, cols], tensor.device())
}

/// Compute the commutator of two matrices: [A, B] = AB - BA
///
/// The commutator measures the extent to which two matrices fail to commute.
///
/// # Arguments
///
/// * `a` - First square matrix (n×n)
/// * `b` - Second square matrix (n×n)
///
/// # Returns
///
/// Commutator matrix [A, B] = AB - BA
///
/// # Properties
///
/// - Anti-symmetric: [A, B] = -[B, A]
/// - [A, A] = 0
/// - Jacobi identity: [A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0
/// - Important in quantum mechanics and Lie algebras
///
/// # Examples
///
/// ```ignore
/// use torsh_linalg::commutator;
/// let a = create_matrix_a()?;
/// let b = create_matrix_b()?;
/// let comm = commutator(&a, &b)?;  // AB - BA
/// ```
pub fn commutator(a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
    validate_square_matrix(a, "Commutator (first matrix)")?;
    validate_square_matrix(b, "Commutator (second matrix)")?;

    if a.shape().dims() != b.shape().dims() {
        return Err(TorshError::InvalidArgument(format!(
            "Commutator requires matrices of same dimensions, got {:?} and {:?}",
            a.shape().dims(),
            b.shape().dims()
        )));
    }

    // [A, B] = AB - BA
    let ab = a.matmul(b)?;
    let ba = b.matmul(a)?;
    ab.sub(&ba)
}

/// Compute the anti-commutator of two matrices: {A, B} = AB + BA
///
/// The anti-commutator is the sum of the two possible matrix products.
///
/// # Arguments
///
/// * `a` - First square matrix (n×n)
/// * `b` - Second square matrix (n×n)
///
/// # Returns
///
/// Anti-commutator matrix {A, B} = AB + BA
///
/// # Properties
///
/// - Symmetric: {A, B} = {B, A}
/// - {A, A} = 2A²
/// - Important in quantum mechanics (fermion algebras)
///
/// # Examples
///
/// ```ignore
/// use torsh_linalg::anticommutator;
/// let a = create_matrix_a()?;
/// let b = create_matrix_b()?;
/// let anticomm = anticommutator(&a, &b)?;  // AB + BA
/// ```
pub fn anticommutator(a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
    validate_square_matrix(a, "Anti-commutator (first matrix)")?;
    validate_square_matrix(b, "Anti-commutator (second matrix)")?;

    if a.shape().dims() != b.shape().dims() {
        return Err(TorshError::InvalidArgument(format!(
            "Anti-commutator requires matrices of same dimensions, got {:?} and {:?}",
            a.shape().dims(),
            b.shape().dims()
        )));
    }

    // {A, B} = AB + BA
    let ab = a.matmul(b)?;
    let ba = b.matmul(a)?;
    ab.add(&ba)
}
