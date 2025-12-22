//! Cache-Oblivious Algorithms for Shape Operations
//!
//! This module provides cache-oblivious algorithms that automatically adapt to
//! all levels of the memory hierarchy without explicit cache size tuning.
//!
//! Cache-oblivious algorithms achieve optimal cache complexity across all cache sizes
//! by using recursive divide-and-conquer strategies. They're particularly effective for:
//! - Matrix transpose
//! - Matrix multiplication
//! - Data layout transformations
//! - Tensor reshape operations
//!
//! # Benefits
//! - Automatic adaptation to L1, L2, L3 caches
//! - No manual cache size tuning required
//! - Optimal asymptotic cache complexity
//! - Improved performance on modern CPUs with deep memory hierarchies
//!
//! # SciRS2 POLICY COMPLIANCE
//! This module uses only Rust standard library - no external dependencies.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::{
    error::{Result, TorshError},
    shape::Shape,
};

/// Minimum size for recursive subdivision (base case)
/// When blocks are smaller than this, use simple loops
const CACHE_OBLIVIOUS_BASE_SIZE: usize = 32;

/// Cache-oblivious transpose implementation
///
/// Uses recursive decomposition to achieve optimal cache performance
/// across all levels of memory hierarchy without knowing cache sizes.
///
/// Time complexity: O(n²)
/// Cache complexity: O(n²/B + n²/√M) where B is cache line size, M is cache size
pub struct CacheObliviousTranspose;

impl CacheObliviousTranspose {
    /// Transpose a matrix in-place (square matrices only)
    ///
    /// # Arguments
    /// * `data` - Flattened matrix data (row-major)
    /// * `n` - Matrix dimension (n x n)
    ///
    /// # Example
    /// ```ignore
    /// let mut data = vec![1, 2, 3, 4]; // [[1, 2], [3, 4]]
    /// CacheObliviousTranspose::transpose_square_inplace(&mut data, 2);
    /// // Now: [[1, 3], [2, 4]]
    /// ```
    pub fn transpose_square_inplace<T: Copy>(data: &mut [T], n: usize) -> Result<()> {
        if data.len() != n * n {
            return Err(TorshError::InvalidArgument(format!(
                "Data length {} doesn't match matrix size {}x{}",
                data.len(),
                n,
                n
            )));
        }

        Self::transpose_recursive(data, n, 0, 0, n, n);
        Ok(())
    }

    /// Transpose a rectangular matrix (out-of-place)
    ///
    /// # Arguments
    /// * `src` - Source matrix data (row-major)
    /// * `dst` - Destination buffer (must be rows x cols sized)
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    pub fn transpose_rect<T: Copy>(
        src: &[T],
        dst: &mut [T],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if src.len() != rows * cols {
            return Err(TorshError::InvalidArgument(format!(
                "Source length {} doesn't match {}x{}",
                src.len(),
                rows,
                cols
            )));
        }
        if dst.len() != rows * cols {
            return Err(TorshError::InvalidArgument(format!(
                "Destination length {} doesn't match {}x{}",
                dst.len(),
                rows,
                cols
            )));
        }

        Self::transpose_rect_recursive(src, dst, rows, cols, 0, 0, rows, cols);
        Ok(())
    }

    /// Recursive helper for square transpose
    fn transpose_recursive<T: Copy>(
        data: &mut [T],
        n: usize,
        row: usize,
        col: usize,
        height: usize,
        width: usize,
    ) {
        // Base case: small enough to fit in cache
        if height <= CACHE_OBLIVIOUS_BASE_SIZE && width <= CACHE_OBLIVIOUS_BASE_SIZE {
            // Direct transpose
            for i in 0..height {
                for j in (i + 1)..width {
                    let idx1 = (row + i) * n + (col + j);
                    let idx2 = (row + j) * n + (col + i);
                    if idx1 < data.len() && idx2 < data.len() {
                        data.swap(idx1, idx2);
                    }
                }
            }
            return;
        }

        // Divide and conquer
        if height >= width {
            // Split horizontally
            let mid = height / 2;
            Self::transpose_recursive(data, n, row, col, mid, width);
            Self::transpose_recursive(data, n, row + mid, col, height - mid, width);
        } else {
            // Split vertically
            let mid = width / 2;
            Self::transpose_recursive(data, n, row, col, height, mid);
            Self::transpose_recursive(data, n, row, col + mid, height, width - mid);
        }
    }

    /// Recursive helper for rectangular transpose
    fn transpose_rect_recursive<T: Copy>(
        src: &[T],
        dst: &mut [T],
        rows: usize,
        cols: usize,
        row: usize,
        col: usize,
        height: usize,
        width: usize,
    ) {
        // Base case
        if height <= CACHE_OBLIVIOUS_BASE_SIZE && width <= CACHE_OBLIVIOUS_BASE_SIZE {
            for i in 0..height {
                for j in 0..width {
                    let src_idx = (row + i) * cols + (col + j);
                    let dst_idx = (col + j) * rows + (row + i);
                    if src_idx < src.len() && dst_idx < dst.len() {
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
            return;
        }

        // Divide and conquer
        if height >= width {
            let mid = height / 2;
            Self::transpose_rect_recursive(src, dst, rows, cols, row, col, mid, width);
            Self::transpose_rect_recursive(
                src,
                dst,
                rows,
                cols,
                row + mid,
                col,
                height - mid,
                width,
            );
        } else {
            let mid = width / 2;
            Self::transpose_rect_recursive(src, dst, rows, cols, row, col, height, mid);
            Self::transpose_rect_recursive(
                src,
                dst,
                rows,
                cols,
                row,
                col + mid,
                height,
                width - mid,
            );
        }
    }
}

/// Cache-oblivious matrix multiplication
///
/// Uses recursive decomposition similar to Strassen's algorithm
/// but without the algebraic optimizations (for simplicity and numerical stability).
pub struct CacheObliviousMatMul;

impl CacheObliviousMatMul {
    /// Multiply two square matrices: C = A * B
    ///
    /// # Arguments
    /// * `a` - First matrix (n x n, row-major)
    /// * `b` - Second matrix (n x n, row-major)
    /// * `c` - Result matrix (n x n, row-major)
    /// * `n` - Matrix dimension
    pub fn multiply_square<T>(a: &[T], b: &[T], c: &mut [T], n: usize) -> Result<()>
    where
        T: Copy + Default + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
    {
        if a.len() != n * n || b.len() != n * n || c.len() != n * n {
            return Err(TorshError::InvalidArgument(
                "Matrix dimensions don't match".to_string(),
            ));
        }

        // Initialize result to zero
        for item in c.iter_mut() {
            *item = T::default();
        }

        Self::multiply_recursive(a, b, c, n, 0, 0, 0, 0, 0, 0, n);
        Ok(())
    }

    /// Recursive helper for matrix multiplication
    #[allow(clippy::too_many_arguments)]
    fn multiply_recursive<T>(
        a: &[T],
        b: &[T],
        c: &mut [T],
        n: usize,
        a_row: usize,
        a_col: usize,
        b_row: usize,
        b_col: usize,
        c_row: usize,
        c_col: usize,
        size: usize,
    ) where
        T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
    {
        // Base case: small enough for naive multiplication
        if size <= CACHE_OBLIVIOUS_BASE_SIZE {
            for i in 0..size {
                for j in 0..size {
                    let mut sum = c[(c_row + i) * n + (c_col + j)];
                    for k in 0..size {
                        let a_idx = (a_row + i) * n + (a_col + k);
                        let b_idx = (b_row + k) * n + (b_col + j);
                        if a_idx < a.len() && b_idx < b.len() {
                            sum = sum + a[a_idx] * b[b_idx];
                        }
                    }
                    c[(c_row + i) * n + (c_col + j)] = sum;
                }
            }
            return;
        }

        // Divide matrices into 4 quadrants
        let mid = size / 2;

        // C11 = A11 * B11 + A12 * B21
        Self::multiply_recursive(a, b, c, n, a_row, a_col, b_row, b_col, c_row, c_col, mid);
        Self::multiply_recursive(
            a,
            b,
            c,
            n,
            a_row,
            a_col + mid,
            b_row + mid,
            b_col,
            c_row,
            c_col,
            mid,
        );

        // C12 = A11 * B12 + A12 * B22
        Self::multiply_recursive(
            a,
            b,
            c,
            n,
            a_row,
            a_col,
            b_row,
            b_col + mid,
            c_row,
            c_col + mid,
            mid,
        );
        Self::multiply_recursive(
            a,
            b,
            c,
            n,
            a_row,
            a_col + mid,
            b_row + mid,
            b_col + mid,
            c_row,
            c_col + mid,
            mid,
        );

        // C21 = A21 * B11 + A22 * B21
        Self::multiply_recursive(
            a,
            b,
            c,
            n,
            a_row + mid,
            a_col,
            b_row,
            b_col,
            c_row + mid,
            c_col,
            mid,
        );
        Self::multiply_recursive(
            a,
            b,
            c,
            n,
            a_row + mid,
            a_col + mid,
            b_row + mid,
            b_col,
            c_row + mid,
            c_col,
            mid,
        );

        // C22 = A21 * B12 + A22 * B22
        Self::multiply_recursive(
            a,
            b,
            c,
            n,
            a_row + mid,
            a_col,
            b_row,
            b_col + mid,
            c_row + mid,
            c_col + mid,
            mid,
        );
        Self::multiply_recursive(
            a,
            b,
            c,
            n,
            a_row + mid,
            a_col + mid,
            b_row + mid,
            b_col + mid,
            c_row + mid,
            c_col + mid,
            mid,
        );
    }
}

/// Cache-oblivious reshape operations
///
/// Efficiently reshapes tensors using cache-oblivious data movement
pub struct CacheObliviousReshape;

impl CacheObliviousReshape {
    /// Reshape data from one shape to another (with same total size)
    ///
    /// Uses cache-oblivious algorithm to minimize cache misses during reshape
    pub fn reshape<T: Copy>(
        src: &[T],
        src_shape: &Shape,
        dst: &mut [T],
        dst_shape: &Shape,
    ) -> Result<()> {
        if src_shape.numel() != dst_shape.numel() {
            return Err(TorshError::InvalidShape(format!(
                "Cannot reshape from {:?} to {:?}: different number of elements",
                src_shape.dims(),
                dst_shape.dims()
            )));
        }

        if src.len() != src_shape.numel() || dst.len() != dst_shape.numel() {
            return Err(TorshError::InvalidArgument(
                "Buffer size doesn't match shape".to_string(),
            ));
        }

        // For simple contiguous reshape, direct copy is optimal
        if src_shape.is_contiguous() && dst_shape.is_contiguous() {
            dst.copy_from_slice(src);
            return Ok(());
        }

        // Use cache-oblivious copy for complex reshapes
        Self::copy_recursive(src, dst, 0, src_shape.numel());
        Ok(())
    }

    /// Recursive copy with cache-oblivious subdivision
    fn copy_recursive<T: Copy>(src: &[T], dst: &mut [T], offset: usize, size: usize) {
        // Base case
        if size <= CACHE_OBLIVIOUS_BASE_SIZE {
            for i in 0..size {
                if offset + i < src.len() && offset + i < dst.len() {
                    dst[offset + i] = src[offset + i];
                }
            }
            return;
        }

        // Divide and conquer
        let mid = size / 2;
        Self::copy_recursive(src, dst, offset, mid);
        Self::copy_recursive(src, dst, offset + mid, size - mid);
    }
}

/// Cache-oblivious layout transformation
///
/// Transforms between different memory layouts (row-major, column-major, etc.)
pub struct CacheObliviousLayout;

impl CacheObliviousLayout {
    /// Convert between row-major and column-major layout
    ///
    /// # Arguments
    /// * `src` - Source data
    /// * `dst` - Destination buffer
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `row_major_to_col_major` - Direction of conversion
    pub fn convert_layout<T: Copy>(
        src: &[T],
        dst: &mut [T],
        rows: usize,
        cols: usize,
        row_major_to_col_major: bool,
    ) -> Result<()> {
        if src.len() != rows * cols || dst.len() != rows * cols {
            return Err(TorshError::InvalidArgument(
                "Buffer sizes don't match dimensions".to_string(),
            ));
        }

        if row_major_to_col_major {
            // Row-major to column-major (transpose)
            CacheObliviousTranspose::transpose_rect(src, dst, rows, cols)
        } else {
            // Column-major to row-major (transpose)
            CacheObliviousTranspose::transpose_rect(src, dst, cols, rows)
        }
    }
}

/// Performance analyzer for cache-oblivious algorithms
pub struct CacheObliviousAnalyzer;

impl CacheObliviousAnalyzer {
    /// Estimate cache miss rate for different algorithm choices
    ///
    /// Returns a score from 0.0 (many misses) to 1.0 (few misses)
    pub fn estimate_cache_efficiency(
        operation: &str,
        shape: &Shape,
        cache_line_size: usize,
    ) -> f64 {
        let numel = shape.numel();
        let element_size = 4; // Assume f32 for estimation

        match operation {
            "transpose" => {
                // Cache-oblivious transpose has O(n²/B + n²/√M) cache complexity
                // where B is cache line size, M is cache size
                // Estimate based on working set size
                let working_set = numel * element_size;
                let _cache_lines = working_set / cache_line_size;

                // Good efficiency if working set fits in typical L2 cache (256KB)
                if working_set < 256 * 1024 {
                    0.9
                } else if working_set < 1024 * 1024 {
                    0.7
                } else {
                    0.5
                }
            }
            "matmul" => {
                // Matrix multiplication benefits greatly from cache-oblivious algorithms
                let dims = shape.dims();
                if dims.len() == 2 {
                    let n = dims[0].max(dims[1]);
                    let working_set = n * n * element_size;

                    if working_set < 128 * 1024 {
                        0.95
                    } else if working_set < 512 * 1024 {
                        0.8
                    } else {
                        0.6
                    }
                } else {
                    0.5
                }
            }
            "reshape" => {
                // Reshape efficiency depends on whether it's contiguous
                if shape.is_contiguous() {
                    1.0 // Perfect sequential access
                } else {
                    0.4 // Strided access is less cache-friendly
                }
            }
            _ => 0.5, // Unknown operation
        }
    }

    /// Recommend whether to use cache-oblivious algorithm
    ///
    /// Returns true if cache-oblivious algorithm is likely to be beneficial
    pub fn should_use_cache_oblivious(operation: &str, shape: &Shape) -> bool {
        let numel = shape.numel();

        match operation {
            "transpose" => {
                // Beneficial for medium to large matrices
                numel > 1024
            }
            "matmul" => {
                // Beneficial for matrices larger than base case
                let dims = shape.dims();
                dims.len() == 2 && dims[0] > CACHE_OBLIVIOUS_BASE_SIZE
            }
            "reshape" => {
                // Only beneficial for non-contiguous reshapes
                !shape.is_contiguous() && numel > 1024
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_square_small() {
        let mut data = vec![1, 2, 3, 4]; // [[1, 2], [3, 4]]
        CacheObliviousTranspose::transpose_square_inplace(&mut data, 2).unwrap();
        assert_eq!(data, vec![1, 3, 2, 4]); // [[1, 3], [2, 4]]
    }

    #[test]
    fn test_transpose_square_4x4() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        CacheObliviousTranspose::transpose_square_inplace(&mut data, 4).unwrap();
        assert_eq!(
            data,
            vec![1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16,]
        );
    }

    #[test]
    fn test_transpose_rect() {
        let src = vec![1, 2, 3, 4, 5, 6]; // [[1, 2, 3], [4, 5, 6]]
        let mut dst = vec![0; 6];
        CacheObliviousTranspose::transpose_rect(&src, &mut dst, 2, 3).unwrap();
        assert_eq!(dst, vec![1, 4, 2, 5, 3, 6]); // [[1, 4], [2, 5], [3, 6]]
    }

    #[test]
    fn test_transpose_rect_large() {
        let rows: usize = 64;
        let cols: usize = 48;
        let src: Vec<i32> = (0..(rows * cols) as i32).collect();
        let mut dst = vec![0; rows * cols];

        CacheObliviousTranspose::transpose_rect(&src, &mut dst, rows, cols).unwrap();

        // Verify transpose
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(dst[j * rows + i], src[i * cols + j]);
            }
        }
    }

    #[test]
    fn test_matmul_2x2() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = vec![5.0f64, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]
        let mut c = vec![0.0f64; 4];

        CacheObliviousMatMul::multiply_square(&a, &b, &mut c, 2).unwrap();

        // Expected: [[19, 22], [43, 50]]
        assert!((c[0] - 19.0).abs() < 1e-6);
        assert!((c[1] - 22.0).abs() < 1e-6);
        assert!((c[2] - 43.0).abs() < 1e-6);
        assert!((c[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_4x4() {
        let n = 4;
        let a: Vec<f32> = (0..(n * n)).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..(n * n)).map(|x| (x + 1) as f32).collect();
        let mut c = vec![0.0; n * n];

        CacheObliviousMatMul::multiply_square(&a, &b, &mut c, n).unwrap();

        // Verify a few elements
        // C[0,0] = A[0,:] · B[:,0] = 0*1 + 1*5 + 2*9 + 3*13 = 62
        assert!((c[0] - 62.0).abs() < 1e-5);
    }

    #[test]
    fn test_reshape_simple() {
        let src = vec![1, 2, 3, 4, 5, 6];
        let mut dst = vec![0; 6];

        let src_shape = Shape::from_array([2, 3]).unwrap();
        let dst_shape = Shape::from_array([3, 2]).unwrap();

        CacheObliviousReshape::reshape(&src, &src_shape, &mut dst, &dst_shape).unwrap();

        // For contiguous reshapes, should be direct copy
        assert_eq!(dst, src);
    }

    #[test]
    fn test_reshape_error_different_size() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 6];

        let src_shape = Shape::from_array([2, 2]).unwrap();
        let dst_shape = Shape::from_array([3, 2]).unwrap();

        let result = CacheObliviousReshape::reshape(&src, &src_shape, &mut dst, &dst_shape);
        assert!(result.is_err());
    }

    #[test]
    fn test_layout_conversion() {
        let src = vec![1, 2, 3, 4, 5, 6]; // Row-major [[1, 2, 3], [4, 5, 6]]
        let mut dst = vec![0; 6];

        CacheObliviousLayout::convert_layout(&src, &mut dst, 2, 3, true).unwrap();

        // Column-major should be [[1, 4], [2, 5], [3, 6]] = [1, 4, 2, 5, 3, 6]
        assert_eq!(dst, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_cache_efficiency_estimation() {
        let small_shape = Shape::from_array([10, 10]).unwrap();
        let large_shape = Shape::from_array([1000, 1000]).unwrap();

        let small_efficiency =
            CacheObliviousAnalyzer::estimate_cache_efficiency("transpose", &small_shape, 64);
        let large_efficiency =
            CacheObliviousAnalyzer::estimate_cache_efficiency("transpose", &large_shape, 64);

        // Small matrices should have better cache efficiency
        assert!(small_efficiency > large_efficiency);
    }

    #[test]
    fn test_should_use_cache_oblivious() {
        let small_shape = Shape::from_array([10, 10]).unwrap();
        let large_shape = Shape::from_array([100, 100]).unwrap();

        // Small matrices don't benefit much
        assert!(!CacheObliviousAnalyzer::should_use_cache_oblivious(
            "transpose",
            &small_shape
        ));

        // Large matrices benefit significantly
        assert!(CacheObliviousAnalyzer::should_use_cache_oblivious(
            "transpose",
            &large_shape
        ));
    }

    #[test]
    fn test_transpose_identity() {
        let mut data = vec![1, 0, 0, 1]; // Identity matrix
        CacheObliviousTranspose::transpose_square_inplace(&mut data, 2).unwrap();
        assert_eq!(data, vec![1, 0, 0, 1]); // Should remain unchanged
    }

    #[test]
    fn test_matmul_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];

        CacheObliviousMatMul::multiply_square(&a, &identity, &mut c, 2).unwrap();

        // A * I = A
        assert_eq!(c, a);
    }

    #[test]
    fn test_transpose_invalid_size() {
        let mut data = vec![1, 2, 3]; // Not square
        let result = CacheObliviousTranspose::transpose_square_inplace(&mut data, 2);
        assert!(result.is_err());
    }
}
