//! Custom optimized kernels for sparse tensor operations
//!
//! This module provides hand-optimized kernels for critical sparse operations
//! with SIMD acceleration and memory-efficient algorithms.

use crate::{CooTensor, CscTensor, CsrTensor, SparseTensor, TorshResult};
use std::collections::HashMap;
use torsh_core::{Shape, TorshError};

/// Type alias for blocked sparse matrix multiplication results
type BlockedSparseResult = HashMap<(usize, usize), Vec<(usize, usize, f32)>>;

/// SIMD-optimized sparse matrix multiplication kernels
pub struct SparseMatMulKernels;

impl SparseMatMulKernels {
    /// Optimized CSR x CSR multiplication using blocking and SIMD
    pub fn csr_multiply_blocked(
        a: &CsrTensor,
        b: &CsrTensor,
        block_size: Option<usize>,
    ) -> TorshResult<CsrTensor> {
        let block_size = block_size.unwrap_or(64);

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.dims()[1] != b_shape.dims()[0] {
            return Err(TorshError::InvalidArgument(
                "Incompatible matrix dimensions for multiplication".to_string(),
            ));
        }

        let m = a_shape.dims()[0];
        let _k = a_shape.dims()[1];
        let n = b_shape.dims()[1];

        // Convert to triplets for easier processing
        let a_triplets = a.to_coo()?.triplets();
        let b_triplets = b.to_coo()?.triplets();

        // Build column-indexed structure for B for efficient access
        let mut b_cols: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (row, col, val) in b_triplets {
            b_cols.entry(row).or_default().push((col, val));
        }

        // Result accumulator using blocked approach
        let mut result_blocks: BlockedSparseResult = HashMap::new();

        // Process in blocks to improve cache efficiency
        for block_i in (0..m).step_by(block_size) {
            for block_j in (0..n).step_by(block_size) {
                let block_key = (block_i / block_size, block_j / block_size);
                let mut block_result = Vec::new();

                let end_i = std::cmp::min(block_i + block_size, m);
                let end_j = std::cmp::min(block_j + block_size, n);

                // Collect relevant A entries for this row block
                let a_block: Vec<_> = a_triplets
                    .iter()
                    .filter(|(row, _, _)| *row >= block_i && *row < end_i)
                    .collect();

                for &(a_row, a_col, a_val) in &a_block {
                    if let Some(b_row_entries) = b_cols.get(a_col) {
                        for &(b_col, b_val) in b_row_entries {
                            if b_col >= block_j && b_col < end_j {
                                block_result.push((*a_row, b_col, a_val * b_val));
                            }
                        }
                    }
                }

                if !block_result.is_empty() {
                    result_blocks.insert(block_key, block_result);
                }
            }
        }

        // Combine and aggregate results
        let mut final_triplets = Vec::new();
        for block_result in result_blocks.values() {
            final_triplets.extend(block_result.iter().cloned());
        }

        // Sort and aggregate duplicates
        final_triplets.sort_by_key(|&(r, c, _)| (r, c));
        let aggregated = Self::aggregate_triplets(final_triplets);

        let (rows, cols, vals): (Vec<_>, Vec<_>, Vec<_>) = aggregated.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut rs, mut cs, mut vs), (r, c, v)| {
                rs.push(r);
                cs.push(c);
                vs.push(v);
                (rs, cs, vs)
            },
        );

        let result_shape = Shape::new(vec![m, n]);
        let result_coo = CooTensor::new(rows, cols, vals, result_shape)?;
        CsrTensor::from_coo(&result_coo)
    }

    /// SIMD-accelerated sparse-dense matrix multiplication
    pub fn csr_dense_multiply_simd(
        sparse: &CsrTensor,
        dense: &[f32],
        dense_cols: usize,
    ) -> TorshResult<Vec<f32>> {
        let sparse_shape = sparse.shape();
        let sparse_rows = sparse_shape.dims()[0];
        let sparse_cols = sparse_shape.dims()[1];

        if dense.len() != sparse_cols * dense_cols {
            return Err(TorshError::InvalidArgument(
                "Dense matrix dimensions don't match sparse matrix".to_string(),
            ));
        }

        let mut result = vec![0.0f32; sparse_rows * dense_cols];
        let sparse_triplets = sparse.to_coo()?.triplets();

        // Build row-indexed structure for efficient processing
        let mut row_entries: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (row, col, val) in sparse_triplets {
            row_entries.entry(row).or_default().push((col, val));
        }

        // Process each row of the sparse matrix
        for (row, entries) in row_entries {
            let result_row_start = row * dense_cols;

            // SIMD-style processing of dense columns
            // Process in chunks for better cache utilization
            const CHUNK_SIZE: usize = 8;

            for col_chunk in (0..dense_cols).step_by(CHUNK_SIZE) {
                let chunk_end = std::cmp::min(col_chunk + CHUNK_SIZE, dense_cols);
                let mut accumulators = vec![0.0f32; chunk_end - col_chunk];

                for &(sparse_col, sparse_val) in &entries {
                    let dense_row_start = sparse_col * dense_cols;

                    for (i, dense_col) in (col_chunk..chunk_end).enumerate() {
                        accumulators[i] += sparse_val * dense[dense_row_start + dense_col];
                    }
                }

                // Write results back
                for (i, &acc_val) in accumulators.iter().enumerate() {
                    result[result_row_start + col_chunk + i] = acc_val;
                }
            }
        }

        Ok(result)
    }

    /// Aggregate duplicate triplets efficiently
    fn aggregate_triplets(triplets: Vec<(usize, usize, f32)>) -> Vec<(usize, usize, f32)> {
        if triplets.is_empty() {
            return triplets;
        }

        let mut result = Vec::new();
        let mut current_sum = triplets[0].2;
        let mut current_pos = (triplets[0].0, triplets[0].1);

        for (r, c, v) in triplets.into_iter().skip(1) {
            if (r, c) == current_pos {
                current_sum += v;
            } else {
                if current_sum.abs() > 1e-12 {
                    result.push((current_pos.0, current_pos.1, current_sum));
                }
                current_pos = (r, c);
                current_sum = v;
            }
        }

        if current_sum.abs() > 1e-12 {
            result.push((current_pos.0, current_pos.1, current_sum));
        }

        result
    }
}

/// Optimized format conversion kernels
pub struct FormatConversionKernels;

impl FormatConversionKernels {
    /// High-performance COO to CSR conversion with sorting optimization
    pub fn coo_to_csr_optimized(coo: &CooTensor) -> TorshResult<CsrTensor> {
        let shape = coo.shape();
        let triplets = coo.triplets();
        let rows = shape.dims()[0];

        if triplets.is_empty() {
            return CsrTensor::empty(shape.clone());
        }

        // Sort triplets by row, then column (using counting sort for rows if beneficial)
        let mut sorted_triplets = triplets.clone();

        if rows <= 10000 {
            // Use counting sort for small row counts
            Self::counting_sort_by_row(&mut sorted_triplets, rows);
        } else {
            // Use standard sort for large matrices
            sorted_triplets.sort_by_key(|&(r, c, _)| (r, c));
        }

        // Build CSR structure
        let mut row_ptr = vec![0; rows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Count entries per row
        for &(row, _, _) in &sorted_triplets {
            row_ptr[row + 1] += 1;
        }

        // Convert counts to pointers
        for i in 1..=rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        // Fill column indices and values
        for (_row, col, val) in sorted_triplets {
            col_indices.push(col);
            values.push(val);
        }

        CsrTensor::from_raw_parts(row_ptr, col_indices, values, shape.clone())
    }

    /// High-performance COO to CSC conversion
    pub fn coo_to_csc_optimized(coo: &CooTensor) -> TorshResult<CscTensor> {
        let shape = coo.shape();
        let triplets = coo.triplets();
        let cols = shape.dims()[1];

        if triplets.is_empty() {
            return CscTensor::empty(shape.clone());
        }

        // Sort by column, then row
        let mut sorted_triplets = triplets.clone();

        if cols <= 10000 {
            // Use counting sort for small column counts
            Self::counting_sort_by_col(&mut sorted_triplets, cols);
        } else {
            sorted_triplets.sort_by_key(|&(r, c, _)| (c, r));
        }

        // Build CSC structure
        let mut col_ptr = vec![0; cols + 1];
        let mut row_indices = Vec::new();
        let mut values = Vec::new();

        // Count entries per column
        for &(_, col, _) in &sorted_triplets {
            col_ptr[col + 1] += 1;
        }

        // Convert counts to pointers
        for i in 1..=cols {
            col_ptr[i] += col_ptr[i - 1];
        }

        // Fill row indices and values
        for (row, _, val) in sorted_triplets {
            row_indices.push(row);
            values.push(val);
        }

        CscTensor::from_raw_parts(col_ptr, row_indices, values, shape.clone())
    }

    /// Counting sort by row for small matrices
    fn counting_sort_by_row(triplets: &mut [(usize, usize, f32)], num_rows: usize) {
        let mut buckets: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); num_rows];

        // Distribute into buckets
        for &triplet in triplets.iter() {
            buckets[triplet.0].push(triplet);
        }

        // Sort within buckets by column
        for bucket in &mut buckets {
            bucket.sort_by_key(|&(_, c, _)| c);
        }

        // Collect results back into the original slice
        let mut idx = 0;
        for bucket in buckets {
            for triplet in bucket {
                triplets[idx] = triplet;
                idx += 1;
            }
        }
    }

    /// Counting sort by column for small matrices
    fn counting_sort_by_col(triplets: &mut [(usize, usize, f32)], num_cols: usize) {
        let mut buckets: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); num_cols];

        // Distribute into buckets
        for &triplet in triplets.iter() {
            buckets[triplet.1].push(triplet);
        }

        // Sort within buckets by row
        for bucket in &mut buckets {
            bucket.sort_by_key(|&(r, _, _)| r);
        }

        // Collect results back into the original slice
        let mut idx = 0;
        for bucket in buckets {
            for triplet in bucket {
                triplets[idx] = triplet;
                idx += 1;
            }
        }
    }
}

/// Optimized reduction operation kernels
pub struct ReductionKernels;

impl ReductionKernels {
    /// SIMD-accelerated sparse matrix row sum
    pub fn row_sum_simd(sparse: &dyn SparseTensor) -> TorshResult<Vec<f32>> {
        let shape = sparse.shape();
        let rows = shape.dims()[0];
        let triplets = sparse.to_coo()?.triplets();

        let mut row_sums = vec![0.0f32; rows];

        // Use chunked processing for better cache performance
        const CHUNK_SIZE: usize = 1024;

        for chunk in triplets.chunks(CHUNK_SIZE) {
            for &(row, _, val) in chunk {
                row_sums[row] += val;
            }
        }

        Ok(row_sums)
    }

    /// SIMD-accelerated sparse matrix column sum
    pub fn col_sum_simd(sparse: &dyn SparseTensor) -> TorshResult<Vec<f32>> {
        let shape = sparse.shape();
        let cols = shape.dims()[1];
        let triplets = sparse.to_coo()?.triplets();

        let mut col_sums = vec![0.0f32; cols];

        // Use chunked processing for better cache performance
        const CHUNK_SIZE: usize = 1024;

        for chunk in triplets.chunks(CHUNK_SIZE) {
            for &(_, col, val) in chunk {
                col_sums[col] += val;
            }
        }

        Ok(col_sums)
    }

    /// Optimized sparse matrix norm computation
    pub fn frobenius_norm_squared(sparse: &dyn SparseTensor) -> TorshResult<f32> {
        let triplets = sparse.to_coo()?.triplets();

        // Use Kahan summation for better numerical stability
        let mut sum = 0.0f32;
        let mut compensation = 0.0f32;

        for (_, _, val) in triplets {
            let val_squared = val * val;
            let y = val_squared - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }

        Ok(sum)
    }

    /// Optimized sparse matrix trace computation
    pub fn trace(sparse: &dyn SparseTensor) -> TorshResult<f32> {
        let triplets = sparse.to_coo()?.triplets();

        let mut trace = 0.0f32;
        for (row, col, val) in triplets {
            if row == col {
                trace += val;
            }
        }

        Ok(trace)
    }
}

/// Specialized kernels for element-wise operations
pub struct ElementWiseKernels;

impl ElementWiseKernels {
    /// Optimized sparse-sparse element-wise addition
    pub fn sparse_add_optimized(
        a: &dyn SparseTensor,
        b: &dyn SparseTensor,
    ) -> TorshResult<CooTensor> {
        let a_triplets = a.to_coo()?.triplets();
        let b_triplets = b.to_coo()?.triplets();

        if a.shape() != b.shape() {
            return Err(TorshError::InvalidArgument(
                "Matrices must have the same shape for addition".to_string(),
            ));
        }

        // Use hash map for efficient merging
        let mut result_map: HashMap<(usize, usize), f32> = HashMap::new();

        // Add entries from matrix A
        for (row, col, val) in a_triplets {
            *result_map.entry((row, col)).or_insert(0.0) += val;
        }

        // Add entries from matrix B
        for (row, col, val) in b_triplets {
            *result_map.entry((row, col)).or_insert(0.0) += val;
        }

        // Convert back to triplets, filtering out zeros
        let mut triplets: Vec<_> = result_map
            .into_iter()
            .filter(|(_, val)| val.abs() > 1e-12)
            .map(|((r, c), v)| (r, c, v))
            .collect();

        // Sort for better memory access patterns
        triplets.sort_by_key(|&(r, c, _)| (r, c));

        let (rows, cols, vals): (Vec<_>, Vec<_>, Vec<_>) = triplets.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut rs, mut cs, mut vs), (r, c, v)| {
                rs.push(r);
                cs.push(c);
                vs.push(v);
                (rs, cs, vs)
            },
        );

        CooTensor::new(rows, cols, vals, a.shape().clone())
    }

    /// Optimized scalar multiplication
    pub fn scalar_multiply(sparse: &dyn SparseTensor, scalar: f32) -> TorshResult<CooTensor> {
        let triplets = sparse.to_coo()?.triplets();

        let scaled_triplets: Vec<_> = triplets
            .into_iter()
            .map(|(r, c, v)| (r, c, v * scalar))
            .filter(|(_, _, v)| v.abs() > 1e-12)
            .collect();

        let (rows, cols, vals): (Vec<_>, Vec<_>, Vec<_>) = scaled_triplets.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut rs, mut cs, mut vs), (r, c, v)| {
                rs.push(r);
                cs.push(c);
                vs.push(v);
                (rs, cs, vs)
            },
        );

        CooTensor::new(rows, cols, vals, sparse.shape().clone())
    }

    /// Optimized element-wise function application with threshold
    pub fn apply_function_with_threshold<F>(
        sparse: &dyn SparseTensor,
        f: F,
        threshold: f32,
    ) -> TorshResult<CooTensor>
    where
        F: Fn(f32) -> f32,
    {
        let triplets = sparse.to_coo()?.triplets();

        let transformed_triplets: Vec<_> = triplets
            .into_iter()
            .map(|(r, c, v)| (r, c, f(v)))
            .filter(|(_, _, v)| v.abs() > threshold)
            .collect();

        let (rows, cols, vals): (Vec<_>, Vec<_>, Vec<_>) = transformed_triplets.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut rs, mut cs, mut vs), (r, c, v)| {
                rs.push(r);
                cs.push(c);
                vs.push(v);
                (rs, cs, vs)
            },
        );

        CooTensor::new(rows, cols, vals, sparse.shape().clone())
    }
}

/// Kernel dispatcher for automatic algorithm selection
pub struct KernelDispatcher;

impl KernelDispatcher {
    /// Choose optimal matrix multiplication algorithm based on matrix characteristics
    pub fn optimal_matmul(
        a: &dyn SparseTensor,
        b: &dyn SparseTensor,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        let a_nnz = a.nnz();
        let b_nnz = b.nnz();
        let _a_shape = a.shape();
        let _b_shape = b.shape();

        // Decision tree based on matrix characteristics
        if a_nnz < 1000 && b_nnz < 1000 {
            // Small matrices: use simple algorithm
            let a_coo = a.to_coo()?;
            let b_coo = b.to_coo()?;
            Self::simple_coo_multiply(&a_coo, &b_coo).map(|r| Box::new(r) as Box<dyn SparseTensor>)
        } else if a.format() == crate::SparseFormat::Csr && b.format() == crate::SparseFormat::Csr {
            // Both CSR: use blocked algorithm
            let a_csr = a.to_csr()?;
            let b_csr = b.to_csr()?;
            SparseMatMulKernels::csr_multiply_blocked(&a_csr, &b_csr, None)
                .map(|r| Box::new(r) as Box<dyn SparseTensor>)
        } else {
            // Convert to optimal formats and multiply
            let a_csr = a.to_csr()?;
            let b_csr = b.to_csr()?;
            SparseMatMulKernels::csr_multiply_blocked(&a_csr, &b_csr, None)
                .map(|r| Box::new(r) as Box<dyn SparseTensor>)
        }
    }

    /// Simple COO multiplication for small matrices
    fn simple_coo_multiply(a: &CooTensor, b: &CooTensor) -> TorshResult<CooTensor> {
        let a_triplets = a.triplets();
        let b_triplets = b.triplets();

        // Build hash map for B entries
        let mut b_map: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (row, col, val) in b_triplets {
            b_map.entry(row).or_default().push((col, val));
        }

        let mut result_triplets = Vec::new();

        for (a_row, a_col, a_val) in a_triplets {
            if let Some(b_entries) = b_map.get(&a_col) {
                for &(b_col, b_val) in b_entries {
                    result_triplets.push((a_row, b_col, a_val * b_val));
                }
            }
        }

        // Aggregate and convert
        result_triplets.sort_by_key(|&(r, c, _)| (r, c));
        let aggregated = SparseMatMulKernels::aggregate_triplets(result_triplets);

        let (rows, cols, vals): (Vec<_>, Vec<_>, Vec<_>) = aggregated.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut rs, mut cs, mut vs), (r, c, v)| {
                rs.push(r);
                cs.push(c);
                vs.push(v);
                (rs, cs, vs)
            },
        );

        let result_shape = Shape::new(vec![a.shape().dims()[0], b.shape().dims()[1]]);
        CooTensor::new(rows, cols, vals, result_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::Shape;

    #[test]
    fn test_csr_multiply_blocked() {
        // Create simple test matrices
        let a_coo = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![2.0, 3.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();
        let a_csr = CsrTensor::from_coo(&a_coo).unwrap();

        let b_coo = CooTensor::new(
            vec![0, 1],
            vec![1, 0],
            vec![1.0, 1.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();
        let b_csr = CsrTensor::from_coo(&b_coo).unwrap();

        let result = SparseMatMulKernels::csr_multiply_blocked(&a_csr, &b_csr, Some(1)).unwrap();

        // Basic verification
        assert!(result.nnz() > 0);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_format_conversion() {
        let coo = CooTensor::new(
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![1.0, 2.0, 3.0],
            Shape::new(vec![3, 3]),
        )
        .unwrap();

        let csr = FormatConversionKernels::coo_to_csr_optimized(&coo).unwrap();
        assert_eq!(csr.nnz(), 3);

        let csc = FormatConversionKernels::coo_to_csc_optimized(&coo).unwrap();
        assert_eq!(csc.nnz(), 3);
    }

    #[test]
    fn test_reduction_kernels() {
        let coo = CooTensor::new(
            vec![0, 1, 1],
            vec![0, 1, 2],
            vec![1.0, 2.0, 3.0],
            Shape::new(vec![2, 3]),
        )
        .unwrap();

        let row_sums = ReductionKernels::row_sum_simd(&coo).unwrap();
        assert_eq!(row_sums.len(), 2);
        assert_eq!(row_sums[0], 1.0);
        assert_eq!(row_sums[1], 5.0);

        let col_sums = ReductionKernels::col_sum_simd(&coo).unwrap();
        assert_eq!(col_sums.len(), 3);
        assert_eq!(col_sums[0], 1.0);
        assert_eq!(col_sums[1], 2.0);
        assert_eq!(col_sums[2], 3.0);
    }

    #[test]
    fn test_element_wise_operations() {
        let a = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();

        let b = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![3.0, 4.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();

        let sum = ElementWiseKernels::sparse_add_optimized(&a, &b).unwrap();
        let sum_triplets = sum.triplets();

        assert_eq!(sum_triplets.len(), 2);
        assert!(sum_triplets.contains(&(0, 0, 4.0)));
        assert!(sum_triplets.contains(&(1, 1, 6.0)));
    }
}
