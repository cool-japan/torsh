//! Comprehensive sparse operations support for ToRSh backends
//!
//! This module provides efficient sparse matrix and tensor operations, including
//! different storage formats and optimized kernels for sparse computations.
//!
//! Supported formats:
//! - COO (Coordinate format)
//! - CSR (Compressed Sparse Row)
//! - CSC (Compressed Sparse Column)
//! - BSR (Block Sparse Row)
//! - Hybrid formats for mixed workloads

use crate::{BackendResult, Device};
use std::collections::HashMap;
use torsh_core::error::TorshError;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// Sparse matrix storage format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Coordinate format (COO) - stores (row, col, value) triplets
    Coo,
    /// Compressed Sparse Row (CSR) format
    Csr,
    /// Compressed Sparse Column (CSC) format
    Csc,
    /// Block Sparse Row (BSR) format - for structured sparsity
    Bsr,
    /// Dense format (for comparison and conversion)
    Dense,
}

/// Sparse matrix structure
#[derive(Debug, Clone)]
pub struct SparseMatrix<T> {
    /// Storage format
    pub format: SparseFormat,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Values of non-zero elements
    pub values: Vec<T>,
    /// Row indices (format-dependent meaning)
    pub row_indices: Vec<usize>,
    /// Column indices (format-dependent meaning)
    pub col_indices: Vec<usize>,
    /// Block size for BSR format
    pub block_size: Option<(usize, usize)>,
}

impl<T> Default for SparseMatrix<T> {
    fn default() -> Self {
        Self {
            format: SparseFormat::Coo,
            rows: 0,
            cols: 0,
            nnz: 0,
            values: Vec::new(),
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            block_size: None,
        }
    }
}

impl<T: Clone + Default + PartialEq> SparseMatrix<T> {
    /// Create a new sparse matrix in COO format
    pub fn new_coo(rows: usize, cols: usize) -> Self {
        Self {
            format: SparseFormat::Coo,
            rows,
            cols,
            nnz: 0,
            values: Vec::new(),
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            block_size: None,
        }
    }

    /// Create a new sparse matrix in CSR format
    pub fn new_csr(rows: usize, cols: usize) -> Self {
        Self {
            format: SparseFormat::Csr,
            rows,
            cols,
            nnz: 0,
            values: Vec::new(),
            row_indices: Vec::with_capacity(rows + 1), // row_ptr array
            col_indices: Vec::new(),
            block_size: None,
        }
    }

    /// Create a new sparse matrix in CSC format
    pub fn new_csc(rows: usize, cols: usize) -> Self {
        Self {
            format: SparseFormat::Csc,
            rows,
            cols,
            nnz: 0,
            values: Vec::new(),
            row_indices: Vec::new(),
            col_indices: Vec::with_capacity(cols + 1), // col_ptr array
            block_size: None,
        }
    }

    /// Insert a value at (row, col) for COO format
    pub fn insert_coo(&mut self, row: usize, col: usize, value: T) -> BackendResult<()> {
        if self.format != SparseFormat::Coo {
            return Err(TorshError::ComputeError(
                "Matrix is not in COO format".to_string(),
            ));
        }

        if row >= self.rows || col >= self.cols {
            return Err(TorshError::ComputeError("Index out of bounds".to_string()));
        }

        // For simplicity, we always append (real implementation would handle duplicates)
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
        self.nnz += 1;

        Ok(())
    }

    /// Convert COO to CSR format
    pub fn to_csr(&self) -> BackendResult<SparseMatrix<T>> {
        if self.format != SparseFormat::Coo {
            return Err(TorshError::ComputeError(
                "Source matrix must be in COO format".to_string(),
            ));
        }

        let mut csr = SparseMatrix::new_csr(self.rows, self.cols);
        csr.nnz = self.nnz;

        if self.nnz == 0 {
            // Initialize empty row_ptr array
            csr.row_indices = vec![0; self.rows + 1];
            return Ok(csr);
        }

        // Count non-zeros per row
        let mut row_counts = vec![0; self.rows];
        for &row in &self.row_indices {
            row_counts[row] += 1;
        }

        // Build row_ptr array (cumulative sum)
        csr.row_indices.push(0);
        for count in row_counts {
            let last = *csr
                .row_indices
                .last()
                .expect("row_indices should not be empty after initial push");
            csr.row_indices.push(last + count);
        }

        // Sort entries by row, then by column
        let mut triplets: Vec<(usize, usize, T)> = self
            .row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
            .map(|((&r, &c), v)| (r, c, v.clone()))
            .collect();

        triplets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Fill values and col_indices
        csr.values.reserve(self.nnz);
        csr.col_indices.reserve(self.nnz);

        for (_, col, value) in triplets {
            csr.col_indices.push(col);
            csr.values.push(value);
        }

        Ok(csr)
    }

    /// Convert COO to CSC format
    pub fn to_csc(&self) -> BackendResult<SparseMatrix<T>> {
        if self.format != SparseFormat::Coo {
            return Err(TorshError::ComputeError(
                "Source matrix must be in COO format".to_string(),
            ));
        }

        let mut csc = SparseMatrix::new_csc(self.rows, self.cols);
        csc.nnz = self.nnz;

        if self.nnz == 0 {
            // Initialize empty col_ptr array
            csc.col_indices = vec![0; self.cols + 1];
            return Ok(csc);
        }

        // Count non-zeros per column
        let mut col_counts = vec![0; self.cols];
        for &col in &self.col_indices {
            col_counts[col] += 1;
        }

        // Build col_ptr array (cumulative sum)
        csc.col_indices.push(0);
        for count in col_counts {
            let last = *csc
                .col_indices
                .last()
                .expect("col_indices should not be empty after initial push");
            csc.col_indices.push(last + count);
        }

        // Sort entries by column, then by row
        let mut triplets: Vec<(usize, usize, T)> = self
            .row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
            .map(|((&r, &c), v)| (r, c, v.clone()))
            .collect();

        triplets.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        // Fill values and row_indices
        csc.values.reserve(self.nnz);
        csc.row_indices.reserve(self.nnz);

        for (row, _, value) in triplets {
            csc.row_indices.push(row);
            csc.values.push(value);
        }

        Ok(csc)
    }

    /// Get sparsity ratio (percentage of non-zero elements)
    pub fn sparsity_ratio(&self) -> f64 {
        if self.rows == 0 || self.cols == 0 {
            return 0.0;
        }
        self.nnz as f64 / (self.rows * self.cols) as f64
    }

    /// Check if the matrix is effectively sparse (< 50% non-zero)
    pub fn is_sparse(&self) -> bool {
        self.sparsity_ratio() < 0.5
    }
}

/// Sparse operations trait for different backends
pub trait SparseOps<T> {
    /// Sparse matrix-vector multiplication: y = A * x
    fn spmv(&self, matrix: &SparseMatrix<T>, x: &[T], y: &mut [T]) -> BackendResult<()>;

    /// Sparse matrix-matrix multiplication: C = A * B
    fn spmm(&self, a: &SparseMatrix<T>, b: &SparseMatrix<T>) -> BackendResult<SparseMatrix<T>>;

    /// Sparse matrix addition: C = A + B
    fn sparse_add(
        &self,
        a: &SparseMatrix<T>,
        b: &SparseMatrix<T>,
    ) -> BackendResult<SparseMatrix<T>>;

    /// Convert sparse matrix to dense format
    fn to_dense(&self, matrix: &SparseMatrix<T>) -> BackendResult<Vec<T>>;

    /// Create sparse matrix from dense format
    fn from_dense(
        &self,
        dense: &[T],
        rows: usize,
        cols: usize,
        threshold: T,
    ) -> BackendResult<SparseMatrix<T>>;

    /// Transpose sparse matrix
    fn transpose(&self, matrix: &SparseMatrix<T>) -> BackendResult<SparseMatrix<T>>;
}

/// Default sparse operations implementation
#[derive(Debug)]
pub struct DefaultSparseOps {
    /// Device for operations
    #[allow(dead_code)]
    device: Device,
    /// Optimization hints
    optimization_hints: SparseOptimizationHints,
}

impl DefaultSparseOps {
    /// Create new sparse operations instance
    pub fn new(device: Device) -> Self {
        Self {
            device,
            optimization_hints: SparseOptimizationHints::default(),
        }
    }

    /// Set optimization hints
    pub fn with_hints(mut self, hints: SparseOptimizationHints) -> Self {
        self.optimization_hints = hints;
        self
    }
}

impl SparseOps<f32> for DefaultSparseOps {
    fn spmv(&self, matrix: &SparseMatrix<f32>, x: &[f32], y: &mut [f32]) -> BackendResult<()> {
        if x.len() != matrix.cols || y.len() != matrix.rows {
            return Err(TorshError::ComputeError("Dimension mismatch".to_string()));
        }

        // Initialize output to zero
        y.fill(0.0);

        match matrix.format {
            SparseFormat::Csr => self.spmv_csr(matrix, x, y),
            SparseFormat::Coo => self.spmv_coo(matrix, x, y),
            SparseFormat::Csc => self.spmv_csc(matrix, x, y),
            _ => Err(TorshError::ComputeError(
                "Unsupported sparse format for SpMV".to_string(),
            )),
        }
    }

    fn spmm(
        &self,
        a: &SparseMatrix<f32>,
        b: &SparseMatrix<f32>,
    ) -> BackendResult<SparseMatrix<f32>> {
        if a.cols != b.rows {
            return Err(TorshError::ComputeError(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        // For simplicity, convert both to CSR format
        let a_csr = if a.format == SparseFormat::Csr {
            a.clone()
        } else {
            a.to_csr()?
        };

        let b_csr = if b.format == SparseFormat::Csr {
            b.clone()
        } else {
            b.to_csr()?
        };

        self.spmm_csr_csr(&a_csr, &b_csr)
    }

    fn sparse_add(
        &self,
        a: &SparseMatrix<f32>,
        b: &SparseMatrix<f32>,
    ) -> BackendResult<SparseMatrix<f32>> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(TorshError::ComputeError(
                "Matrix dimensions must match for addition".to_string(),
            ));
        }

        // Convert both to COO format for easier addition
        let a_coo = if a.format == SparseFormat::Coo {
            a.clone()
        } else {
            // For now, only support COO to CSR conversion
            return Err(TorshError::ComputeError(
                "Sparse addition requires COO format".to_string(),
            ));
        };

        let b_coo = if b.format == SparseFormat::Coo {
            b.clone()
        } else {
            return Err(TorshError::ComputeError(
                "Sparse addition requires COO format".to_string(),
            ));
        };

        self.sparse_add_coo(&a_coo, &b_coo)
    }

    fn to_dense(&self, matrix: &SparseMatrix<f32>) -> BackendResult<Vec<f32>> {
        let mut dense = vec![0.0; matrix.rows * matrix.cols];

        match matrix.format {
            SparseFormat::Coo => {
                for i in 0..matrix.nnz {
                    let row = matrix.row_indices[i];
                    let col = matrix.col_indices[i];
                    let val = matrix.values[i];
                    dense[row * matrix.cols + col] = val;
                }
            }
            SparseFormat::Csr => {
                for row in 0..matrix.rows {
                    let start = matrix.row_indices[row];
                    let end = matrix.row_indices[row + 1];
                    for idx in start..end {
                        let col = matrix.col_indices[idx];
                        let val = matrix.values[idx];
                        dense[row * matrix.cols + col] = val;
                    }
                }
            }
            SparseFormat::Csc => {
                for col in 0..matrix.cols {
                    let start = matrix.col_indices[col];
                    let end = matrix.col_indices[col + 1];
                    for idx in start..end {
                        let row = matrix.row_indices[idx];
                        let val = matrix.values[idx];
                        dense[row * matrix.cols + col] = val;
                    }
                }
            }
            _ => {
                return Err(TorshError::ComputeError(
                    "Unsupported format for dense conversion".to_string(),
                ))
            }
        }

        Ok(dense)
    }

    fn from_dense(
        &self,
        dense: &[f32],
        rows: usize,
        cols: usize,
        threshold: f32,
    ) -> BackendResult<SparseMatrix<f32>> {
        if dense.len() != rows * cols {
            return Err(TorshError::ComputeError(
                "Dense array size doesn't match dimensions".to_string(),
            ));
        }

        let mut sparse = SparseMatrix::new_coo(rows, cols);

        for row in 0..rows {
            for col in 0..cols {
                let val = dense[row * cols + col];
                if val.abs() > threshold {
                    sparse.insert_coo(row, col, val)?;
                }
            }
        }

        Ok(sparse)
    }

    fn transpose(&self, matrix: &SparseMatrix<f32>) -> BackendResult<SparseMatrix<f32>> {
        match matrix.format {
            SparseFormat::Coo => {
                let mut transposed = SparseMatrix::new_coo(matrix.cols, matrix.rows);
                transposed.nnz = matrix.nnz;

                // Swap row and column indices
                transposed.row_indices = matrix.col_indices.clone();
                transposed.col_indices = matrix.row_indices.clone();
                transposed.values = matrix.values.clone();

                Ok(transposed)
            }
            SparseFormat::Csr => {
                // CSR transpose becomes CSC
                let mut transposed = SparseMatrix::new_csc(matrix.cols, matrix.rows);
                transposed.nnz = matrix.nnz;
                transposed.values = matrix.values.clone();
                transposed.row_indices = matrix.col_indices.clone();
                transposed.col_indices = matrix.row_indices.clone();
                Ok(transposed)
            }
            SparseFormat::Csc => {
                // CSC transpose becomes CSR
                let mut transposed = SparseMatrix::new_csr(matrix.cols, matrix.rows);
                transposed.nnz = matrix.nnz;
                transposed.values = matrix.values.clone();
                transposed.row_indices = matrix.col_indices.clone();
                transposed.col_indices = matrix.row_indices.clone();
                Ok(transposed)
            }
            _ => Err(TorshError::ComputeError(
                "Unsupported format for transpose".to_string(),
            )),
        }
    }
}

impl DefaultSparseOps {
    /// CSR format SpMV implementation
    fn spmv_csr(&self, matrix: &SparseMatrix<f32>, x: &[f32], y: &mut [f32]) -> BackendResult<()> {
        for row in 0..matrix.rows {
            let start = matrix.row_indices[row];
            let end = matrix.row_indices[row + 1];
            let mut sum = 0.0;

            for idx in start..end {
                let col = matrix.col_indices[idx];
                let val = matrix.values[idx];
                sum += val * x[col];
            }

            y[row] = sum;
        }
        Ok(())
    }

    /// COO format SpMV implementation
    fn spmv_coo(&self, matrix: &SparseMatrix<f32>, x: &[f32], y: &mut [f32]) -> BackendResult<()> {
        for i in 0..matrix.nnz {
            let row = matrix.row_indices[i];
            let col = matrix.col_indices[i];
            let val = matrix.values[i];
            y[row] += val * x[col];
        }
        Ok(())
    }

    /// CSC format SpMV implementation
    fn spmv_csc(&self, matrix: &SparseMatrix<f32>, x: &[f32], y: &mut [f32]) -> BackendResult<()> {
        for col in 0..matrix.cols {
            let start = matrix.col_indices[col];
            let end = matrix.col_indices[col + 1];
            let x_val = x[col];

            for idx in start..end {
                let row = matrix.row_indices[idx];
                let val = matrix.values[idx];
                y[row] += val * x_val;
            }
        }
        Ok(())
    }

    /// CSR x CSR matrix multiplication
    fn spmm_csr_csr(
        &self,
        a: &SparseMatrix<f32>,
        b: &SparseMatrix<f32>,
    ) -> BackendResult<SparseMatrix<f32>> {
        // This is a simplified implementation
        // Real implementation would use more sophisticated algorithms
        let mut result = SparseMatrix::new_coo(a.rows, b.cols);

        for row_a in 0..a.rows {
            let start_a = a.row_indices[row_a];
            let end_a = a.row_indices[row_a + 1];

            for idx_a in start_a..end_a {
                let col_a = a.col_indices[idx_a];
                let val_a = a.values[idx_a];

                // col_a is the row in matrix B
                let start_b = b.row_indices[col_a];
                let end_b = b.row_indices[col_a + 1];

                for idx_b in start_b..end_b {
                    let col_b = b.col_indices[idx_b];
                    let val_b = b.values[idx_b];

                    let product = val_a * val_b;
                    result.insert_coo(row_a, col_b, product)?;
                }
            }
        }

        Ok(result)
    }

    /// COO format sparse addition
    fn sparse_add_coo(
        &self,
        a: &SparseMatrix<f32>,
        b: &SparseMatrix<f32>,
    ) -> BackendResult<SparseMatrix<f32>> {
        let mut result = SparseMatrix::new_coo(a.rows, a.cols);

        // Use a hashmap to combine duplicate entries
        let mut entries: HashMap<(usize, usize), f32> = HashMap::new();

        // Add entries from matrix A
        for i in 0..a.nnz {
            let key = (a.row_indices[i], a.col_indices[i]);
            *entries.entry(key).or_insert(0.0) += a.values[i];
        }

        // Add entries from matrix B
        for i in 0..b.nnz {
            let key = (b.row_indices[i], b.col_indices[i]);
            *entries.entry(key).or_insert(0.0) += b.values[i];
        }

        // Convert back to COO format
        for ((row, col), value) in entries {
            if value != 0.0 {
                result.insert_coo(row, col, value)?;
            }
        }

        Ok(result)
    }
}

impl<T: Clone + Default + PartialEq> SparseMatrix<T> {
    /// Create BSR (Block Sparse Row) matrix from COO
    pub fn to_bsr(&self, block_size: (usize, usize)) -> BackendResult<SparseMatrix<T>> {
        if self.format != SparseFormat::Coo {
            return Err(TorshError::ComputeError(
                "Source matrix must be in COO format".to_string(),
            ));
        }

        let (block_rows, block_cols) = block_size;
        if block_rows == 0 || block_cols == 0 {
            return Err(TorshError::ComputeError(
                "Block size must be positive".to_string(),
            ));
        }

        // Calculate number of block rows and columns
        let num_block_rows = (self.rows + block_rows - 1) / block_rows;
        let _num_block_cols = (self.cols + block_cols - 1) / block_cols;

        let mut bsr = SparseMatrix {
            format: SparseFormat::Bsr,
            rows: self.rows,
            cols: self.cols,
            nnz: 0,
            values: Vec::new(),
            row_indices: vec![0; num_block_rows + 1], // block row pointers
            col_indices: Vec::new(),                  // block column indices
            block_size: Some(block_size),
        };

        // Group non-zeros by blocks
        let mut blocks: HashMap<(usize, usize), Vec<T>> = HashMap::new();

        for i in 0..self.nnz {
            let row = self.row_indices[i];
            let col = self.col_indices[i];
            let val = self.values[i].clone();

            let block_row = row / block_rows;
            let block_col = col / block_cols;
            let in_block_row = row % block_rows;
            let in_block_col = col % block_cols;

            let block_entry = blocks
                .entry((block_row, block_col))
                .or_insert_with(|| vec![T::default(); block_rows * block_cols]);
            block_entry[in_block_row * block_cols + in_block_col] = val;
        }

        // Convert to BSR format
        let mut sorted_blocks: Vec<_> = blocks.into_iter().collect();
        sorted_blocks.sort_by_key(|&((br, bc), _)| (br, bc));

        let mut current_block_row = 0;
        for ((block_row, block_col), block_values) in sorted_blocks {
            // Update row pointers
            while current_block_row < block_row {
                current_block_row += 1;
                bsr.row_indices[current_block_row] = bsr.col_indices.len();
            }

            // Add block
            bsr.col_indices.push(block_col);
            bsr.values.extend(block_values);
            bsr.nnz += 1; // Number of blocks, not individual elements
        }

        // Fill remaining row pointers
        let final_ptr = bsr.col_indices.len();
        for i in (current_block_row + 1)..=num_block_rows {
            bsr.row_indices[i] = final_ptr;
        }

        Ok(bsr)
    }

    /// Optimize matrix structure by removing explicit zeros and sorting
    pub fn optimize(&mut self) -> BackendResult<()> {
        match self.format {
            SparseFormat::Coo => {
                // Remove explicit zeros and sort by (row, col)
                let mut triplets: Vec<(usize, usize, T)> = (0..self.nnz)
                    .filter_map(|i| {
                        let val = &self.values[i];
                        if *val != T::default() {
                            Some((self.row_indices[i], self.col_indices[i], val.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();

                triplets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

                // Rebuild arrays
                self.nnz = triplets.len();
                self.row_indices.clear();
                self.col_indices.clear();
                self.values.clear();

                for (row, col, val) in triplets {
                    self.row_indices.push(row);
                    self.col_indices.push(col);
                    self.values.push(val);
                }
            }
            SparseFormat::Csr | SparseFormat::Csc => {
                // Remove explicit zeros while maintaining format
                let mut new_values = Vec::new();
                let mut new_col_indices = Vec::new();
                let mut new_row_pointers = vec![0];

                let num_rows = if self.format == SparseFormat::Csr {
                    self.rows
                } else {
                    self.cols
                };

                for row in 0..num_rows {
                    let start = self.row_indices[row];
                    let end = self.row_indices[row + 1];

                    for idx in start..end {
                        if self.values[idx] != T::default() {
                            new_values.push(self.values[idx].clone());
                            new_col_indices.push(self.col_indices[idx]);
                        }
                    }
                    new_row_pointers.push(new_values.len());
                }

                self.values = new_values;
                self.col_indices = new_col_indices;
                self.row_indices = new_row_pointers;
                self.nnz = self.values.len();
            }
            _ => {
                return Err(TorshError::ComputeError(
                    "Optimization not supported for this format".to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Get matrix statistics for performance analysis
    pub fn statistics(&self) -> SparseMatrixStatistics {
        let mut max_row_nnz = 0;
        let mut min_row_nnz = usize::MAX;
        let mut row_nnz_variance = 0.0;

        match self.format {
            SparseFormat::Csr => {
                let mut row_counts = Vec::new();
                for row in 0..self.rows {
                    let count = self.row_indices[row + 1] - self.row_indices[row];
                    row_counts.push(count);
                    max_row_nnz = max_row_nnz.max(count);
                    min_row_nnz = min_row_nnz.min(count);
                }

                let mean = row_counts.iter().sum::<usize>() as f64 / row_counts.len() as f64;
                row_nnz_variance = row_counts
                    .iter()
                    .map(|&x| (x as f64 - mean).powi(2))
                    .sum::<f64>()
                    / row_counts.len() as f64;
            }
            SparseFormat::Coo => {
                let mut row_counts = vec![0; self.rows];
                for &row in &self.row_indices {
                    row_counts[row] += 1;
                }
                max_row_nnz = *row_counts.iter().max().unwrap_or(&0);
                min_row_nnz = *row_counts.iter().min().unwrap_or(&0);

                let mean = self.nnz as f64 / self.rows as f64;
                row_nnz_variance = row_counts
                    .iter()
                    .map(|&x| (x as f64 - mean).powi(2))
                    .sum::<f64>()
                    / self.rows as f64;
            }
            _ => {
                // For other formats, provide basic stats
                min_row_nnz = if self.nnz == 0 { 0 } else { 1 };
            }
        }

        SparseMatrixStatistics {
            format: self.format,
            rows: self.rows,
            cols: self.cols,
            nnz: self.nnz,
            sparsity_ratio: self.sparsity_ratio(),
            max_row_nnz,
            min_row_nnz,
            row_nnz_variance,
            memory_usage: self.estimated_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes
    fn estimated_memory_usage(&self) -> usize {
        SparseFormatConverter::estimate_memory_usage(self.rows, self.cols, self.nnz, self.format)
    }
}

/// Optimization hints for sparse operations
#[derive(Debug, Clone)]
pub struct SparseOptimizationHints {
    /// Prefer memory efficiency over speed
    pub memory_efficient: bool,
    /// Use parallel processing when available
    pub use_parallel: bool,
    /// Expected sparsity level (0.0 to 1.0)
    pub expected_sparsity: f64,
    /// Block size for BSR format operations
    pub block_size: Option<(usize, usize)>,
    /// Cache block size for tiled operations
    pub cache_block_size: usize,
}

impl Default for SparseOptimizationHints {
    fn default() -> Self {
        Self {
            memory_efficient: true,
            use_parallel: true,
            expected_sparsity: 0.1, // 10% non-zero by default
            block_size: None,
            cache_block_size: 64,
        }
    }
}

/// Sparse format conversion utilities
pub struct SparseFormatConverter;

impl SparseFormatConverter {
    /// Automatically choose the best format based on matrix properties
    pub fn choose_optimal_format<T>(
        _matrix: &SparseMatrix<T>,
        operation: SparseOperation,
    ) -> SparseFormat {
        match operation {
            SparseOperation::SpMV => {
                // CSR is typically best for SpMV
                SparseFormat::Csr
            }
            SparseOperation::SpMM => {
                // CSR x CSC is often efficient for SpMM
                SparseFormat::Csr
            }
            SparseOperation::Addition => {
                // COO is easiest for addition
                SparseFormat::Coo
            }
            SparseOperation::Transpose => {
                // COO is format-agnostic for transpose
                SparseFormat::Coo
            }
            SparseOperation::Iterative => {
                // CSR is good for iterative methods
                SparseFormat::Csr
            }
        }
    }

    /// Get memory usage estimate for different formats
    pub fn estimate_memory_usage(
        rows: usize,
        cols: usize,
        nnz: usize,
        format: SparseFormat,
    ) -> usize {
        match format {
            SparseFormat::Coo => {
                // (row_idx, col_idx, value) for each non-zero
                nnz * (std::mem::size_of::<usize>() * 2 + std::mem::size_of::<f32>())
            }
            SparseFormat::Csr => {
                // row_ptr array + col_indices + values
                (rows + 1) * std::mem::size_of::<usize>()
                    + nnz * (std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
            }
            SparseFormat::Csc => {
                // col_ptr array + row_indices + values
                (cols + 1) * std::mem::size_of::<usize>()
                    + nnz * (std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
            }
            SparseFormat::Dense => rows * cols * std::mem::size_of::<f32>(),
            _ => nnz * std::mem::size_of::<f32>() * 3, // Conservative estimate
        }
    }
}

/// Types of sparse operations for format optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseOperation {
    /// Sparse matrix-vector multiplication
    SpMV,
    /// Sparse matrix-matrix multiplication
    SpMM,
    /// Matrix addition
    Addition,
    /// Matrix transpose
    Transpose,
    /// Iterative solver operations
    Iterative,
}

/// Statistics for sparse matrix analysis and optimization
#[derive(Debug, Clone)]
pub struct SparseMatrixStatistics {
    /// Matrix storage format
    pub format: SparseFormat,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Sparsity ratio (0.0 to 1.0)
    pub sparsity_ratio: f64,
    /// Maximum non-zeros in any row
    pub max_row_nnz: usize,
    /// Minimum non-zeros in any row
    pub min_row_nnz: usize,
    /// Variance in row non-zero counts
    pub row_nnz_variance: f64,
    /// Estimated memory usage in bytes
    pub memory_usage: usize,
}

impl SparseMatrixStatistics {
    /// Check if matrix structure is well-balanced
    pub fn is_well_balanced(&self) -> bool {
        if self.rows == 0 || self.nnz == 0 {
            return true;
        }

        let avg_nnz_per_row = self.nnz as f64 / self.rows as f64;
        let balance_ratio = self.max_row_nnz as f64 / avg_nnz_per_row.max(1.0);

        // Consider well-balanced if max row doesn't have more than 3x average
        balance_ratio < 3.0
    }

    /// Get recommended operations based on matrix characteristics
    pub fn recommended_operations(&self) -> Vec<&'static str> {
        let mut recommendations = Vec::new();

        if self.sparsity_ratio < 0.1 {
            recommendations.push("Very sparse - excellent for sparse algorithms");
        } else if self.sparsity_ratio > 0.5 {
            recommendations.push("Dense - consider dense algorithms");
        }

        if !self.is_well_balanced() {
            recommendations.push("Unbalanced structure - consider load balancing");
        }

        match self.format {
            SparseFormat::Coo => {
                recommendations.push("COO format - good for construction and element access")
            }
            SparseFormat::Csr => {
                recommendations.push("CSR format - optimal for SpMV and most algorithms")
            }
            SparseFormat::Csc => recommendations.push("CSC format - good for transpose operations"),
            SparseFormat::Bsr => {
                recommendations.push("BSR format - optimal for block-structured sparsity")
            }
            SparseFormat::Dense => recommendations.push("Dense format - use dense linear algebra"),
        }

        recommendations
    }
}

/// Hardware-accelerated sparse operations (backend-specific implementations)
pub trait HardwareSparseOps<T>: SparseOps<T> {
    /// Get hardware acceleration capabilities
    fn acceleration_capabilities(&self) -> SparseAccelerationCapabilities;

    /// Batched sparse matrix-vector multiplication
    fn batch_spmv(
        &self,
        matrices: &[&SparseMatrix<T>],
        vectors: &[&[T]],
        results: &mut [&mut [T]],
    ) -> BackendResult<()>;

    /// Fused sparse operations (e.g., SpMV + vector operations)
    fn fused_spmv_add(
        &self,
        matrix: &SparseMatrix<T>,
        x: &[T],
        y: &[T],
        result: &mut [T],
        alpha: T,
        beta: T,
    ) -> BackendResult<()>;

    /// Sparse iterative solver operations
    fn iterative_solve(
        &self,
        matrix: &SparseMatrix<T>,
        b: &[T],
        x0: &[T],
        method: IterativeMethod,
        tolerance: f64,
        max_iterations: usize,
    ) -> BackendResult<SolverResult<T>>;
}

/// Hardware acceleration capabilities for sparse operations
#[derive(Debug, Clone)]
pub struct SparseAccelerationCapabilities {
    /// SIMD vector instructions available
    pub simd_width: usize,
    /// GPU acceleration available
    pub gpu_acceleration: bool,
    /// Specialized sparse hardware (e.g., tensor cores)
    pub specialized_hardware: bool,
    /// Multi-threading support
    pub parallel_execution: bool,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f32,
}

/// Iterative solver methods for sparse linear systems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterativeMethod {
    /// Conjugate Gradient (for symmetric positive definite)
    ConjugateGradient,
    /// BiCGStab (for general systems)
    BiCGStab,
    /// GMRES (for general systems)
    GMRES,
    /// Jacobi iteration
    Jacobi,
    /// Gauss-Seidel iteration
    GaussSeidel,
}

/// Result from iterative solver
#[derive(Debug, Clone)]
pub struct SolverResult<T> {
    /// Solution vector
    pub solution: Vec<T>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: f64,
    /// Whether solver converged
    pub converged: bool,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Advanced sparse operations with optimization and acceleration
#[derive(Debug)]
pub struct AdvancedSparseOps {
    base_ops: DefaultSparseOps,
    acceleration_caps: SparseAccelerationCapabilities,
    performance_cache: HashMap<String, f64>, // Cache for operation timings
}

impl AdvancedSparseOps {
    /// Create new advanced sparse operations instance
    pub fn new(device: Device) -> Self {
        let acceleration_caps = Self::detect_acceleration_capabilities(&device);
        let base_ops = DefaultSparseOps::new(device);

        Self {
            base_ops,
            acceleration_caps,
            performance_cache: HashMap::new(),
        }
    }

    /// Detect hardware acceleration capabilities
    fn detect_acceleration_capabilities(device: &Device) -> SparseAccelerationCapabilities {
        SparseAccelerationCapabilities {
            simd_width: if cfg!(target_arch = "x86_64") { 8 } else { 4 }, // AVX-256 vs NEON-128
            gpu_acceleration: device.device_type() != torsh_core::device::DeviceType::Cpu,
            specialized_hardware: false, // Would be detected based on device
            parallel_execution: true,
            memory_bandwidth: if device.device_type() == torsh_core::device::DeviceType::Cpu {
                50.0
            } else {
                500.0
            },
        }
    }

    /// Optimized sparse matrix-vector multiplication with acceleration
    pub fn optimized_spmv(
        &mut self,
        matrix: &SparseMatrix<f32>,
        x: &[f32],
        y: &mut [f32],
    ) -> BackendResult<()> {
        let _cache_key = format!(
            "spmv_{}_{}_{}_{}",
            matrix.format as u8, matrix.rows, matrix.cols, matrix.nnz
        );

        // Use parallel execution if available and matrix is large enough
        if self.acceleration_caps.parallel_execution && matrix.nnz > 10000 {
            self.parallel_spmv(matrix, x, y)
        } else if self.acceleration_caps.simd_width > 1 {
            self.simd_spmv(matrix, x, y)
        } else {
            self.base_ops.spmv(matrix, x, y)
        }
    }

    /// Parallel sparse matrix-vector multiplication
    fn parallel_spmv(
        &self,
        matrix: &SparseMatrix<f32>,
        x: &[f32],
        y: &mut [f32],
    ) -> BackendResult<()> {
        match matrix.format {
            SparseFormat::Csr => {
                // ✅ SciRS2 POLICY: Use scirs2_core::parallel_ops instead of direct rayon
                use scirs2_core::parallel_ops::*;

                // Parallel iteration over rows
                let row_chunks: Vec<_> = (0..matrix.rows).collect();
                let chunk_size = (matrix.rows + current_num_threads() - 1) / current_num_threads();

                row_chunks.par_chunks(chunk_size).for_each(|chunk| {
                    for &row in chunk {
                        let start = matrix.row_indices[row];
                        let end = matrix.row_indices[row + 1];
                        let mut sum = 0.0;

                        for idx in start..end {
                            let col = matrix.col_indices[idx];
                            let val = matrix.values[idx];
                            sum += val * x[col];
                        }

                        // Safe because each thread works on disjoint rows
                        unsafe {
                            let y_ptr = y.as_ptr() as *mut f32;
                            *y_ptr.add(row) = sum;
                        }
                    }
                });

                Ok(())
            }
            _ => self.base_ops.spmv(matrix, x, y), // Fall back to sequential for other formats
        }
    }

    /// SIMD-accelerated sparse matrix-vector multiplication
    fn simd_spmv(&self, matrix: &SparseMatrix<f32>, x: &[f32], y: &mut [f32]) -> BackendResult<()> {
        // Placeholder for SIMD implementation
        // Real implementation would use architecture-specific SIMD intrinsics
        self.base_ops.spmv(matrix, x, y)
    }

    /// Adaptive format selection based on operation and matrix characteristics
    pub fn adaptive_format_conversion(
        &self,
        matrix: &SparseMatrix<f32>,
        target_operation: SparseOperation,
    ) -> BackendResult<SparseMatrix<f32>> {
        let stats = matrix.statistics();

        let optimal_format = if stats.is_well_balanced() {
            match target_operation {
                SparseOperation::SpMV => SparseFormat::Csr,
                SparseOperation::SpMM => SparseFormat::Csr,
                SparseOperation::Addition => SparseFormat::Coo,
                SparseOperation::Transpose => SparseFormat::Coo,
                SparseOperation::Iterative => SparseFormat::Csr,
            }
        } else {
            // For unbalanced matrices, prefer more flexible formats
            match target_operation {
                SparseOperation::SpMV if stats.max_row_nnz > stats.nnz / 10 => SparseFormat::Coo, // Very unbalanced
                _ => SparseFormatConverter::choose_optimal_format(matrix, target_operation),
            }
        };

        if matrix.format == optimal_format {
            Ok(matrix.clone())
        } else {
            match optimal_format {
                SparseFormat::Csr => matrix.to_csr(),
                SparseFormat::Csc => matrix.to_csc(),
                SparseFormat::Bsr => {
                    let block_size = (8, 8); // Default block size
                    matrix.to_bsr(block_size)
                }
                _ => Ok(matrix.clone()),
            }
        }
    }

    /// Benchmark and cache operation performance
    pub fn benchmark_operation(&mut self, operation: &str, matrix: &SparseMatrix<f32>) -> f64 {
        let cache_key = format!(
            "{}_{}_{}_{}",
            operation, matrix.rows, matrix.cols, matrix.nnz
        );

        if let Some(&cached_time) = self.performance_cache.get(&cache_key) {
            return cached_time;
        }

        // Simplified benchmarking - real implementation would use precise timing
        let estimated_time = match operation {
            "spmv" => {
                (matrix.nnz as f64 * 2.0) / (self.acceleration_caps.memory_bandwidth as f64 * 1e9)
            }
            "spmm" => {
                (matrix.nnz as f64 * matrix.cols as f64 * 2.0)
                    / (self.acceleration_caps.memory_bandwidth as f64 * 1e9)
            }
            _ => 0.001, // Default 1ms
        };

        self.performance_cache.insert(cache_key, estimated_time);
        estimated_time
    }
}

/// Sparse linear algebra utilities
pub struct SparseLinAlg;

impl SparseLinAlg {
    /// Compute sparse matrix norm (Frobenius norm)
    pub fn frobenius_norm<T>(matrix: &SparseMatrix<T>) -> f64
    where
        T: Clone + Default + PartialEq,
        f64: From<T>,
    {
        let mut sum = 0.0;
        for value in &matrix.values {
            let val: f64 = value.clone().into();
            sum += val * val;
        }
        sum.sqrt()
    }

    /// Check if sparse matrix is symmetric
    pub fn is_symmetric(matrix: &SparseMatrix<f32>, tolerance: f32) -> BackendResult<bool> {
        if matrix.rows != matrix.cols {
            return Ok(false);
        }

        // Convert to COO for easier comparison
        let coo = if matrix.format == SparseFormat::Coo {
            matrix.clone()
        } else {
            return Err(TorshError::ComputeError(
                "Symmetry check requires COO format".to_string(),
            ));
        };

        // Create a map of (row, col) -> value for quick lookup
        let mut entries: HashMap<(usize, usize), f32> = HashMap::new();
        for i in 0..coo.nnz {
            entries.insert((coo.row_indices[i], coo.col_indices[i]), coo.values[i]);
        }

        // Check symmetry
        for ((row, col), &value) in &entries {
            if let Some(&transpose_value) = entries.get(&(*col, *row)) {
                if (value - transpose_value).abs() > tolerance {
                    return Ok(false);
                }
            } else if value.abs() > tolerance {
                // Non-zero element with no transpose counterpart
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Extract diagonal of sparse matrix
    pub fn diagonal<T: Clone + Default>(matrix: &SparseMatrix<T>) -> Vec<T> {
        let mut diag = vec![T::default(); matrix.rows.min(matrix.cols)];

        match matrix.format {
            SparseFormat::Coo => {
                for i in 0..matrix.nnz {
                    let row = matrix.row_indices[i];
                    let col = matrix.col_indices[i];
                    if row == col && row < diag.len() {
                        diag[row] = matrix.values[i].clone();
                    }
                }
            }
            SparseFormat::Csr => {
                for row in 0..matrix.rows.min(diag.len()) {
                    let start = matrix.row_indices[row];
                    let end = matrix.row_indices[row + 1];

                    for idx in start..end {
                        let col = matrix.col_indices[idx];
                        if col == row {
                            diag[row] = matrix.values[idx].clone();
                            break;
                        } else if col > row {
                            break; // Assuming sorted columns
                        }
                    }
                }
            }
            _ => {
                // For other formats, convert to COO first (simplified)
                // Real implementation would handle each format optimally
            }
        }

        diag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_creation() {
        let mut matrix = SparseMatrix::<f32>::new_coo(3, 3);
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.nnz, 0);
        assert_eq!(matrix.format, SparseFormat::Coo);

        // Insert some values
        matrix.insert_coo(0, 0, 1.0).unwrap();
        matrix.insert_coo(1, 1, 2.0).unwrap();
        matrix.insert_coo(2, 2, 3.0).unwrap();

        assert_eq!(matrix.nnz, 3);
        assert_eq!(matrix.sparsity_ratio(), 3.0 / 9.0);
        assert!(matrix.is_sparse());
    }

    #[test]
    fn test_coo_to_csr_conversion() {
        let mut coo = SparseMatrix::<f32>::new_coo(3, 3);
        coo.insert_coo(0, 0, 1.0).unwrap();
        coo.insert_coo(0, 2, 2.0).unwrap();
        coo.insert_coo(1, 1, 3.0).unwrap();
        coo.insert_coo(2, 0, 4.0).unwrap();
        coo.insert_coo(2, 2, 5.0).unwrap();

        let csr = coo.to_csr().unwrap();
        assert_eq!(csr.format, SparseFormat::Csr);
        assert_eq!(csr.nnz, 5);

        // Check row_ptr array: should be [0, 2, 3, 5]
        assert_eq!(csr.row_indices, vec![0, 2, 3, 5]);
    }

    #[test]
    fn test_sparse_spmv() {
        let device = Device::cpu().unwrap();
        let sparse_ops = DefaultSparseOps::new(device);

        // Create a simple 3x3 matrix
        let mut matrix = SparseMatrix::<f32>::new_coo(3, 3);
        matrix.insert_coo(0, 0, 2.0).unwrap();
        matrix.insert_coo(1, 1, 3.0).unwrap();
        matrix.insert_coo(2, 2, 4.0).unwrap();

        // Convert to CSR for SpMV
        let csr_matrix = matrix.to_csr().unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];

        sparse_ops.spmv(&csr_matrix, &x, &mut y).unwrap();

        assert_eq!(y, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_sparse_to_dense() {
        let device = Device::cpu().unwrap();
        let sparse_ops = DefaultSparseOps::new(device);

        let mut matrix = SparseMatrix::<f32>::new_coo(2, 2);
        matrix.insert_coo(0, 0, 1.0).unwrap();
        matrix.insert_coo(1, 1, 2.0).unwrap();

        let dense = sparse_ops.to_dense(&matrix).unwrap();
        assert_eq!(dense, vec![1.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_sparse_from_dense() {
        let device = Device::cpu().unwrap();
        let sparse_ops = DefaultSparseOps::new(device);

        let dense = vec![1.0, 0.0, 0.0, 2.0];
        let sparse = sparse_ops.from_dense(&dense, 2, 2, 0.1).unwrap();

        assert_eq!(sparse.nnz, 2);
        assert_eq!(sparse.sparsity_ratio(), 0.5);
    }

    #[test]
    fn test_sparse_transpose() {
        let device = Device::cpu().unwrap();
        let sparse_ops = DefaultSparseOps::new(device);

        let mut matrix = SparseMatrix::<f32>::new_coo(2, 3);
        matrix.insert_coo(0, 1, 1.0).unwrap();
        matrix.insert_coo(1, 2, 2.0).unwrap();

        let transposed = sparse_ops.transpose(&matrix).unwrap();

        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 2);
        assert_eq!(transposed.nnz, 2);

        // Check that indices are swapped
        assert_eq!(transposed.row_indices, vec![1, 2]); // original col_indices
        assert_eq!(transposed.col_indices, vec![0, 1]); // original row_indices
    }

    #[test]
    fn test_memory_usage_estimation() {
        let rows = 1000;
        let cols = 1000;
        let nnz = 10000; // 1% sparsity

        let coo_memory =
            SparseFormatConverter::estimate_memory_usage(rows, cols, nnz, SparseFormat::Coo);
        let csr_memory =
            SparseFormatConverter::estimate_memory_usage(rows, cols, nnz, SparseFormat::Csr);
        let dense_memory =
            SparseFormatConverter::estimate_memory_usage(rows, cols, nnz, SparseFormat::Dense);

        // Dense should use much more memory
        assert!(dense_memory > coo_memory);
        assert!(dense_memory > csr_memory);

        // CSR should be slightly more memory efficient than COO for this case
        assert!(csr_memory < coo_memory);
    }

    #[test]
    fn test_format_selection() {
        let matrix = SparseMatrix::<f32>::new_coo(100, 100);

        let spmv_format =
            SparseFormatConverter::choose_optimal_format(&matrix, SparseOperation::SpMV);
        assert_eq!(spmv_format, SparseFormat::Csr);

        let add_format =
            SparseFormatConverter::choose_optimal_format(&matrix, SparseOperation::Addition);
        assert_eq!(add_format, SparseFormat::Coo);
    }
}
