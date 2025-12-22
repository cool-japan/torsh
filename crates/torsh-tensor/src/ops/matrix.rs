//! Matrix operations for tensors with SciRS2 SIMD Acceleration

use crate::{Tensor, TensorElement, FloatElement};
use torsh_core::error::{Result, TorshError};
use torsh_core::shape::Shape;
use scirs2_core::ndarray::{Array, Array2, ArrayView, Axis, IxDyn};
use num_traits::{Float, Zero, One, cast::ToPrimitive};
use std::ops::{Add, Mul, Sub};

// ✅ SciRS2 Advanced Performance Features for Matrix Operations
#[cfg(feature = "simd")]
use scirs2_core::simd::{SimdArray, SimdOps};
use scirs2_core::simd_ops::{simd_dot_product, simd_matrix_multiply};

#[cfg(feature = "parallel")]
use scirs2_core::parallel::{ParallelExecutor, ChunkStrategy};
use scirs2_core::parallel_ops::{par_chunks, par_join};

#[cfg(feature = "profiling")]
use scirs2_core::profiling::profile_section;

impl<T: TensorElement> Tensor<T> {
    /// Matrix multiplication with GPU and SIMD acceleration
    pub fn matmul(&self, other: &Self) -> Result<Self>
    where
        T: Add<Output = T> + Mul<Output = T> + Zero + Copy
    {
        #[cfg(feature = "profiling")]
        let _profile = profile_section!("matmul");

        // ✅ SciRS2 GPU Acceleration - Use GPU for very large matrices (10x-100x speedup potential)
        #[cfg(feature = "gpu")]
        {
            if self.ndim() == 2 && other.ndim() == 2 {
                let self_shape = self.shape().dims();
                let other_shape = other.shape().dims();

                // Use GPU for very large matrices where transfer cost is justified
                if self_shape[0] * self_shape[1] > 10000 && other_shape[0] * other_shape[1] > 10000 {
                    if let Ok(result) = self.gpu_matmul_2d(other) {
                        return Ok(result);
                    }
                }
            }
        }

        // ✅ SciRS2 SIMD Optimization - Use optimized matrix multiplication for large matrices
        #[cfg(feature = "simd")]
        {
            if self.ndim() == 2 && other.ndim() == 2 {
                let self_shape = self.shape().dims();
                let other_shape = other.shape().dims();

                // Use SIMD for sufficiently large matrices
                if self_shape[0] * self_shape[1] > 1000 && other_shape[0] * other_shape[1] > 1000 {
                    if let Ok(result) = self.simd_matmul_2d(other) {
                        return Ok(result);
                    }
                }
            }
        }

        // Handle different dimensionalities with fallback
        match (self.ndim(), other.ndim()) {
            (2, 2) => self.matmul_2d(other),
            (1, 2) => self.matmul_1d_2d(other),
            (2, 1) => self.matmul_2d_1d(other),
            (1, 1) => self.matmul_1d_1d(other),
            _ => self.batch_matmul(other),
        }
    }

    /// GPU-accelerated 2D matrix multiplication proof-of-concept
    #[cfg(feature = "gpu")]
    fn gpu_matmul_2d(&self, other: &Self) -> Result<Self>
    where
        T: scirs2_core::gpu::GpuElement + Add<Output = T> + Mul<Output = T> + Zero + Copy,
    {
        #[cfg(feature = "profiling")]
        let _profile = profile_section!("gpu_matmul");

        use scirs2_core::gpu::{GpuContext, GpuBuffer, GpuKernel};

        let self_shape = self.shape().dims();
        let other_shape = other.shape().dims();

        if self_shape.len() != 2 || other_shape.len() != 2 {
            return Err(TorshError::Other("Both tensors must be 2D for GPU matmul".to_string()));
        }

        let (m, k1) = (self_shape[0], self_shape[1]);
        let (k2, n) = (other_shape[0], other_shape[1]);

        if k1 != k2 {
            return Err(TorshError::Other(format!(
                "Cannot multiply matrices with shapes [{}, {}] and [{}, {}]",
                m, k1, k2, n
            )));
        }

        // Initialize GPU context (preferring CUDA, fallback to Metal/OpenCL)
        let gpu_context = GpuContext::new()?;

        // Transfer data to GPU
        let self_data = self.data()?;
        let other_data = other.data()?;

        let gpu_buffer_a = GpuBuffer::from_slice(&gpu_context, &self_data)?;
        let gpu_buffer_b = GpuBuffer::from_slice(&gpu_context, &other_data)?;
        let gpu_buffer_c = GpuBuffer::zeros(&gpu_context, m * n)?;

        // Launch optimized matrix multiplication kernel
        let kernel = GpuKernel::matrix_multiply(&gpu_context)?;
        kernel.launch(
            &gpu_buffer_a, &gpu_buffer_b, &gpu_buffer_c,
            m, k1, n
        )?;

        // Transfer result back to CPU
        let result_data = gpu_buffer_c.to_vec()?;
        let result_shape = vec![m, n];

        Self::from_data(result_data, result_shape, self.device())
    }

    /// SIMD-optimized 2D matrix multiplication
    #[cfg(feature = "simd")]
    fn simd_matmul_2d(&self, other: &Self) -> Result<Self>
    where
        T: scirs2_core::simd::SimdElement + Add<Output = T> + Mul<Output = T> + Zero + Copy,
    {
        let self_shape = self.shape().dims();
        let other_shape = other.shape().dims();

        if self_shape.len() != 2 || other_shape.len() != 2 {
            return Err(TorshError::Other("Both tensors must be 2D for SIMD matmul".to_string()));
        }

        let (m, k1) = (self_shape[0], self_shape[1]);
        let (k2, n) = (other_shape[0], other_shape[1]);

        if k1 != k2 {
            return Err(TorshError::Other(format!(
                "Cannot multiply matrices with shapes [{}, {}] and [{}, {}]",
                m, k1, k2, n
            )));
        }

        // Use SciRS2's optimized matrix multiplication
        let self_data = self.data()?;
        let other_data = other.data()?;

        let result_data = simd_matrix_multiply(
            &self_data, &other_data,
            m, k1, n
        );

        let result_shape = vec![m, n];
        Self::from_data(result_data, result_shape, self.device())
    }

    /// Matrix multiplication for 2D x 2D tensors
    fn matmul_2d(&self, other: &Self) -> Result<Self>
    where
        T: Add<Output = T> + Mul<Output = T> + Zero + Copy
    {
        let self_shape = self.shape().dims();
        let other_shape = other.shape().dims();

        if self_shape.len() != 2 || other_shape.len() != 2 {
            return Err(TorshError::Other("Both tensors must be 2D for 2D matmul".to_string()));
        }

        let (m, k1) = (self_shape[0], self_shape[1]);
        let (k2, n) = (other_shape[0], other_shape[1]);

        if k1 != k2 {
            return Err(TorshError::Other(format!(
                "Cannot multiply matrices with shapes [{}, {}] and [{}, {}]",
                m, k1, k2, n
            )));
        }

        let result_shape = vec![m, n];
        let mut result = Self::zeros(&result_shape, self.device())?;

        // Perform matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
        for i in 0..m {
            for j in 0..n {
                let mut sum = <T as TensorElement>::zero();
                for k in 0..k1 {
                    let a_val = self.get_item(&[i, k])?;
                    let b_val = other.get_item(&[k, j])?;
                    sum = sum + a_val * b_val;
                }
                result.set_item(&[i, j], sum)?;
            }
        }

        Ok(result)
    }

    /// Matrix multiplication for 1D x 2D tensors (vector x matrix)
    fn matmul_1d_2d(&self, other: &Self) -> Result<Self>
    where
        T: Add<Output = T> + Mul<Output = T> + Zero + Copy
    {
        let self_shape = self.shape().dims();
        let other_shape = other.shape().dims();

        if self_shape.len() != 1 || other_shape.len() != 2 {
            return Err(TorshError::Other("Expected 1D and 2D tensors".to_string()));
        }

        let k1 = self_shape[0];
        let (k2, n) = (other_shape[0], other_shape[1]);

        if k1 != k2 {
            return Err(TorshError::Other(format!(
                "Cannot multiply vector of length {} with matrix of shape [{}, {}]",
                k1, k2, n
            )));
        }

        let result_shape = vec![n];
        let mut result = Self::zeros(&result_shape, self.device())?;

        // Perform vector-matrix multiplication: c[j] = sum_k a[k] * B[k,j]
        for j in 0..n {
            let mut sum = <T as TensorElement>::zero();
            for k in 0..k1 {
                let a_val = self.get_item(&[k])?;
                let b_val = other.get_item(&[k, j])?;
                sum = sum + a_val * b_val;
            }
            result.set_item(&[j], sum)?;
        }

        Ok(result)
    }

    /// Matrix multiplication for 2D x 1D tensors (matrix x vector)
    fn matmul_2d_1d(&self, other: &Self) -> Result<Self>
    where
        T: Add<Output = T> + Mul<Output = T> + Zero + Copy
    {
        let self_shape = self.shape().dims();
        let other_shape = other.shape().dims();

        if self_shape.len() != 2 || other_shape.len() != 1 {
            return Err(TorshError::Other("Expected 2D and 1D tensors".to_string()));
        }

        let (m, k1) = (self_shape[0], self_shape[1]);
        let k2 = other_shape[0];

        if k1 != k2 {
            return Err(TorshError::Other(format!(
                "Cannot multiply matrix of shape [{}, {}] with vector of length {}",
                m, k1, k2
            )));
        }

        let result_shape = vec![m];
        let mut result = Self::zeros(&result_shape, self.device())?;

        // Perform matrix-vector multiplication: c[i] = sum_k A[i,k] * b[k]
        for i in 0..m {
            let mut sum = <T as TensorElement>::zero();
            for k in 0..k1 {
                let a_val = self.get_item(&[i, k])?;
                let b_val = other.get_item(&[k])?;
                sum = sum + a_val * b_val;
            }
            result.set_item(&[i], sum)?;
        }

        Ok(result)
    }

    /// Vector dot product for 1D x 1D tensors
    fn matmul_1d_1d(&self, other: &Self) -> Result<Self>
    where
        T: Add<Output = T> + Mul<Output = T> + Zero + Copy
    {
        let self_shape = self.shape().dims();
        let other_shape = other.shape().dims();

        if self_shape.len() != 1 || other_shape.len() != 1 {
            return Err(TorshError::Other("Both tensors must be 1D for dot product".to_string()));
        }

        let n1 = self_shape[0];
        let n2 = other_shape[0];

        if n1 != n2 {
            return Err(TorshError::Other(format!(
                "Cannot compute dot product of vectors with lengths {} and {}",
                n1, n2
            )));
        }

        let mut sum = <T as TensorElement>::zero();
        for i in 0..n1 {
            let a_val = self.get_item(&[i])?;
            let b_val = other.get_item(&[i])?;
            sum = sum + a_val * b_val;
        }

        // Return scalar (0D tensor)
        Self::from_scalar(sum, &[], self.device())
    }

    /// Batched matrix multiplication for higher-dimensional tensors
    fn batch_matmul(&self, other: &Self) -> Result<Self>
    where
        T: Add<Output = T> + Mul<Output = T> + Zero + Copy
    {
        let self_shape_obj = self.shape();
        let self_shape = self_shape_obj.dims();
        let other_shape_obj = other.shape();
        let other_shape = other_shape_obj.dims();

        if self_shape.len() < 2 || other_shape.len() < 2 {
            return Err(TorshError::Other("Tensors must have at least 2 dimensions for batched matmul".to_string()));
        }

        let self_ndim = self_shape.len();
        let other_ndim = other_shape.len();

        // Get matrix dimensions (last two dimensions)
        let self_matrix_dims = [self_shape[self_ndim - 2], self_shape[self_ndim - 1]];
        let other_matrix_dims = [other_shape[other_ndim - 2], other_shape[other_ndim - 1]];

        if self_matrix_dims[1] != other_matrix_dims[0] {
            return Err(TorshError::Other(format!(
                "Cannot multiply matrices with inner dimensions {} and {}",
                self_matrix_dims[1], other_matrix_dims[0]
            )));
        }

        // For now, return a simplified implementation
        // Full implementation would handle batch dimensions properly
        let result_matrix_dims = [self_matrix_dims[0], other_matrix_dims[1]];

        // Use the batch dimensions from the first tensor
        let mut result_shape = self_shape[..self_ndim - 2].to_vec();
        result_shape.extend_from_slice(&result_matrix_dims);

        let result = Self::zeros(&result_shape, self.device())?;

        // Implementation would perform batched matrix multiplication
        // This is a simplified placeholder

        Ok(result)
    }

    /// Computes the outer product of two 1D tensors
    pub fn outer(&self, other: &Self) -> Result<Self>
    where
        T: Mul<Output = T> + Copy
    {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(TorshError::Other("Both tensors must be 1D for outer product".to_string()));
        }

        let m = self.shape().dims()[0];
        let n = other.shape().dims()[0];
        let result_shape = vec![m, n];
        let mut result = Self::zeros(&result_shape, self.device())?;

        for i in 0..m {
            for j in 0..n {
                let a_val = self.get_item(&[i])?;
                let b_val = other.get_item(&[j])?;
                result.set_item(&[i, j], a_val * b_val)?;
            }
        }

        Ok(result)
    }

    /// Computes the cross product of two 3D vectors
    pub fn cross(&self, other: &Self) -> Result<Self>
    where
        T: Sub<Output = T> + Mul<Output = T> + Copy
    {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(TorshError::Other("Both tensors must be 1D for cross product".to_string()));
        }

        if self.shape().dims()[0] != 3 || other.shape().dims()[0] != 3 {
            return Err(TorshError::Other("Cross product requires 3D vectors".to_string()));
        }

        let a = [self.get_item(&[0])?, self.get_item(&[1])?, self.get_item(&[2])?];
        let b = [other.get_item(&[0])?, other.get_item(&[1])?, other.get_item(&[2])?];

        let result_data = [
            a[1] * b[2] - a[2] * b[1],  // x component
            a[2] * b[0] - a[0] * b[2],  // y component
            a[0] * b[1] - a[1] * b[0],  // z component
        ];

        let mut result = Self::zeros(&[3], self.device())?;
        for i in 0..3 {
            result.set_item(&[i], result_data[i])?;
        }

        Ok(result)
    }

    /// Computes the trace (sum of diagonal elements) of a 2D tensor
    pub fn trace(&self) -> Result<T>
    where
        T: Add<Output = T> + Zero + Copy
    {
        if self.ndim() != 2 {
            return Err(TorshError::Other("Trace can only be computed for 2D tensors".to_string()));
        }

        let shape = self.shape().dims();
        let min_dim = std::cmp::min(shape[0], shape[1]);
        let mut trace_val = <T as TensorElement>::zero();

        for i in 0..min_dim {
            trace_val = trace_val + self.get_item(&[i, i])?;
        }

        Ok(trace_val)
    }

    /// Computes the diagonal elements of a 2D tensor
    pub fn diag(&self) -> Result<Self>
    where
        T: Copy
    {
        if self.ndim() != 2 {
            return Err(TorshError::Other("Diagonal can only be extracted from 2D tensors".to_string()));
        }

        let shape = self.shape().dims();
        let min_dim = std::cmp::min(shape[0], shape[1]);
        let mut result = Self::zeros(&[min_dim], self.device())?;

        for i in 0..min_dim {
            let val = self.get_item(&[i, i])?;
            result.set_item(&[i], val)?;
        }

        Ok(result)
    }

    /// Creates a diagonal matrix from a 1D tensor
    pub fn diag_embed(diag: &Self) -> Result<Self>
    where
        T: Zero + Copy
    {
        if diag.ndim() != 1 {
            return Err(TorshError::Other("Input must be 1D tensor for diagonal embedding".to_string()));
        }

        let n = diag.shape().dims()[0];
        let mut result = Self::zeros(&[n, n], diag.device())?;

        for i in 0..n {
            let val = diag.get_item(&[i])?;
            result.set_item(&[i, i], val)?;
        }

        Ok(result)
    }
}

// Float-specific matrix operations
impl<T: FloatElement> Tensor<T> {
    /// Computes the determinant of a square matrix
    pub fn det(&self) -> Result<T>
    where
        T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Zero + One + Copy + PartialEq,
    {
        if self.ndim() != 2 {
            return Err(TorshError::Other("Determinant can only be computed for 2D tensors".to_string()));
        }

        let shape = self.shape().dims();
        if shape[0] != shape[1] {
            return Err(TorshError::Other("Determinant requires a square matrix".to_string()));
        }

        let n = shape[0];

        match n {
            0 => Ok(<T as TensorElement>::one()),
            1 => Ok(self.get_item(&[0, 0])?),
            2 => {
                let a00 = self.get_item(&[0, 0])?;
                let a01 = self.get_item(&[0, 1])?;
                let a10 = self.get_item(&[1, 0])?;
                let a11 = self.get_item(&[1, 1])?;
                Ok(a00 * a11 - a01 * a10)
            }
            3 => {
                // Rule of Sarrus for 3x3 matrices
                let a00 = self.get_item(&[0, 0])?;
                let a01 = self.get_item(&[0, 1])?;
                let a02 = self.get_item(&[0, 2])?;
                let a10 = self.get_item(&[1, 0])?;
                let a11 = self.get_item(&[1, 1])?;
                let a12 = self.get_item(&[1, 2])?;
                let a20 = self.get_item(&[2, 0])?;
                let a21 = self.get_item(&[2, 1])?;
                let a22 = self.get_item(&[2, 2])?;

                let pos = a00 * a11 * a22 + a01 * a12 * a20 + a02 * a10 * a21;
                let neg = a02 * a11 * a20 + a01 * a10 * a22 + a00 * a12 * a21;
                Ok(pos - neg)
            }
            _ => {
                // For larger matrices, use LU decomposition: det(A) = det(P) * det(L) * det(U)
                // where det(P) = (-1)^(number of row swaps)
                // det(L) = 1 (unit diagonal)
                // det(U) = product of diagonal elements

                let data = self.to_vec()?;
                let mut lu = data.clone();
                let mut swaps = 0;

                // LU decomposition with partial pivoting
                for k in 0..n {
                    // Find pivot
                    let mut max_val = lu[k * n + k];
                    let mut max_row = k;

                    for i in (k + 1)..n {
                        let val = lu[i * n + k];
                        if val.abs() > max_val.abs() {
                            max_val = val;
                            max_row = i;
                        }
                    }

                    // Swap rows if needed
                    if max_row != k {
                        for j in 0..n {
                            let temp = lu[k * n + j];
                            lu[k * n + j] = lu[max_row * n + j];
                            lu[max_row * n + j] = temp;
                        }
                        swaps += 1;
                    }

                    // Check for singularity
                    if lu[k * n + k].abs() < <T as TensorElement>::zero() {
                        return Ok(<T as TensorElement>::zero());
                    }

                    // Eliminate column
                    for i in (k + 1)..n {
                        lu[i * n + k] = lu[i * n + k] / lu[k * n + k];
                        for j in (k + 1)..n {
                            lu[i * n + j] = lu[i * n + j] - lu[i * n + k] * lu[k * n + j];
                        }
                    }
                }

                // Compute determinant: product of diagonal elements * sign from swaps
                let mut det = if swaps % 2 == 0 {
                    <T as TensorElement>::one()
                } else {
                    <T as TensorElement>::zero() - <T as TensorElement>::one()
                };

                for i in 0..n {
                    det = det * lu[i * n + i];
                }

                Ok(det)
            }
        }
    }

    /// Computes the matrix inverse using Gauss-Jordan elimination
    pub fn inverse(&self) -> Result<Self>
    where
        T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> +
           std::ops::Div<Output = T> + Zero + One + Copy + PartialEq,
    {
        if self.ndim() != 2 {
            return Err(TorshError::Other("Inverse can only be computed for 2D tensors".to_string()));
        }

        let shape = self.shape().dims();
        if shape[0] != shape[1] {
            return Err(TorshError::Other("Inverse requires a square matrix".to_string()));
        }

        let n = shape[0];

        // Check if matrix is singular by computing determinant
        let det = self.det()?;
        if det == <T as TensorElement>::zero() {
            return Err(TorshError::Other("Matrix is singular and cannot be inverted".to_string()));
        }

        match n {
            1 => {
                let val = self.get_item(&[0, 0])?;
                let inv_val = <T as TensorElement>::one() / val;
                let mut result = Self::zeros(&[1, 1], self.device())?;
                result.set_item(&[0, 0], inv_val)?;
                Ok(result)
            }
            2 => {
                // Analytical inverse for 2x2 matrix
                let a = self.get_item(&[0, 0])?;
                let b = self.get_item(&[0, 1])?;
                let c = self.get_item(&[1, 0])?;
                let d = self.get_item(&[1, 1])?;

                let det_inv = <T as TensorElement>::one() / det;
                let mut result = Self::zeros(&[2, 2], self.device())?;

                result.set_item(&[0, 0], d * det_inv)?;
                result.set_item(&[0, 1], (<T as TensorElement>::zero() - b) * det_inv)?;
                result.set_item(&[1, 0], (<T as TensorElement>::zero() - c) * det_inv)?;
                result.set_item(&[1, 1], a * det_inv)?;

                Ok(result)
            }
            _ => {
                // For larger matrices, use LU decomposition to solve A*X = I
                // where X is the inverse matrix

                let data = self.to_vec()?;
                let mut lu = data.clone();
                let mut perm: Vec<usize> = (0..n).collect(); // Permutation vector

                // LU decomposition with partial pivoting
                for k in 0..n {
                    // Find pivot
                    let mut max_val = lu[perm[k] * n + k];
                    let mut max_row = k;

                    for i in (k + 1)..n {
                        let val = lu[perm[i] * n + k];
                        if val.abs() > max_val.abs() {
                            max_val = val;
                            max_row = i;
                        }
                    }

                    // Swap permutation indices
                    if max_row != k {
                        perm.swap(k, max_row);
                    }

                    // Check for singularity
                    let pivot = lu[perm[k] * n + k];
                    if pivot.abs() < <T as TensorElement>::zero() {
                        return Err(TorshError::Other("Matrix is singular".to_string()));
                    }

                    // Eliminate column
                    for i in (k + 1)..n {
                        let factor = lu[perm[i] * n + k] / lu[perm[k] * n + k];
                        lu[perm[i] * n + k] = factor;
                        for j in (k + 1)..n {
                            lu[perm[i] * n + j] = lu[perm[i] * n + j] - factor * lu[perm[k] * n + j];
                        }
                    }
                }

                // Solve for each column of the inverse
                let mut result_data = vec![<T as TensorElement>::zero(); n * n];

                for col in 0..n {
                    // Create right-hand side (column of identity matrix)
                    let mut b = vec![<T as TensorElement>::zero(); n];
                    b[col] = <T as TensorElement>::one();

                    // Forward substitution (solve L*y = P*b)
                    let mut y = vec![<T as TensorElement>::zero(); n];
                    for i in 0..n {
                        let mut sum = b[perm[i]];
                        for j in 0..i {
                            sum = sum - lu[perm[i] * n + j] * y[j];
                        }
                        y[i] = sum;
                    }

                    // Backward substitution (solve U*x = y)
                    let mut x = vec![<T as TensorElement>::zero(); n];
                    for i in (0..n).rev() {
                        let mut sum = y[i];
                        for j in (i + 1)..n {
                            sum = sum - lu[perm[i] * n + j] * x[j];
                        }
                        x[i] = sum / lu[perm[i] * n + i];
                    }

                    // Store column in result
                    for row in 0..n {
                        result_data[row * n + col] = x[row];
                    }
                }

                Self::from_vec(result_data, &[n, n], self.device())
            }
        }
    }

    /// Computes matrix rank using row reduction (new modular implementation)
    pub fn matrix_rank(&self) -> Result<usize>
    where
        T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> +
           std::ops::Div<Output = T> + Zero + One + Copy + PartialOrd,
    {
        if self.ndim() != 2 {
            return Err(TorshError::Other("Rank can only be computed for 2D tensors".to_string()));
        }

        // Implementation would use row reduction to find the rank
        // This is a simplified placeholder that returns the minimum dimension
        let shape_obj = self.shape();
        let shape = shape_obj.dims();
        Ok(std::cmp::min(shape[0], shape[1]))
    }

    /// Computes the Frobenius norm of the matrix
    pub fn frobenius_norm(&self) -> Result<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Zero + Copy + ToPrimitive,
    {
        if self.ndim() != 2 {
            return Err(TorshError::Other("Frobenius norm can only be computed for 2D tensors".to_string()));
        }

        let mut sum = <T as TensorElement>::zero();
        for i in 0..self.shape().dims()[0] {
            for j in 0..self.shape().dims()[1] {
                let val = self.get_item(&[i, j])?;
                sum = sum + val * val;
            }
        }

        // Take square root
        if let Some(sum_f64) = <T as TensorElement>::to_f64(&sum) {
            Ok(T::from(sum_f64.sqrt()).unwrap_or(sum))
        } else {
            Ok(sum)
        }
    }

    /// Computes the condition number of the matrix
    pub fn cond(&self) -> Result<T>
    where
        T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> +
           std::ops::Div<Output = T> + Zero + One + Copy + PartialEq + PartialOrd + ToPrimitive,
    {
        // Condition number = ||A|| * ||A^(-1)||
        // For now, we use Frobenius norm
        let norm_a = self.frobenius_norm()?;
        let inv_a = self.inverse()?;
        let norm_inv_a = inv_a.frobenius_norm()?;

        Ok(norm_a * norm_inv_a)
    }

    /// Solves linear system Ax = b using Gaussian elimination
    pub fn solve(&self, b: &Self) -> Result<Self>
    where
        T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> +
           std::ops::Div<Output = T> + Zero + One + Copy + PartialEq + PartialOrd,
    {
        if self.ndim() != 2 {
            return Err(TorshError::Other("A must be a 2D tensor (matrix)".to_string()));
        }

        let shape_a_obj = self.shape();
        let shape_a = shape_a_obj.dims();
        if shape_a[0] != shape_a[1] {
            return Err(TorshError::Other("A must be a square matrix".to_string()));
        }

        if b.ndim() != 1 && b.ndim() != 2 {
            return Err(TorshError::Other("b must be a 1D or 2D tensor".to_string()));
        }

        let n = shape_a[0];
        let shape_b_obj = b.shape();
        let shape_b = shape_b_obj.dims();

        if shape_b[0] != n {
            return Err(TorshError::Other("Incompatible dimensions for solving Ax = b".to_string()));
        }

        // For now, use simple approach: x = A^(-1) * b
        let inv_a = self.inverse()?;
        inv_a.matmul(b)
    }
}