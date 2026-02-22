//! CSR (Compressed Sparse Row) sparse tensor format

use crate::{CooTensor, SparseFormat, SparseTensor, TorshResult};
use torsh_core::{device::DeviceType, DType, Shape, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

/// CSR (Compressed Sparse Row) format sparse tensor
#[derive(Debug)]
pub struct CsrTensor {
    /// Row pointers (size: rows + 1)
    row_ptr: Vec<usize>,
    /// Column indices
    col_indices: Vec<usize>,
    /// Non-zero values
    values: Vec<f32>,
    /// Shape of the tensor
    shape: Shape,
    /// Data type
    dtype: DType,
    /// Device
    device: DeviceType,
}

impl CsrTensor {
    /// Create a new CSR tensor
    pub fn new(
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f32>,
        shape: Shape,
    ) -> TorshResult<Self> {
        // Validate inputs
        if col_indices.len() != values.len() {
            return Err(TorshError::InvalidArgument(
                "Column indices and values must have the same length".to_string(),
            ));
        }

        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "CSR format currently only supports 2D tensors".to_string(),
            ));
        }

        let rows = shape.dims()[0];
        if row_ptr.len() != rows + 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Row pointer length must be rows + 1, got {} for {} rows",
                row_ptr.len(),
                rows
            )));
        }

        // Validate column indices are within bounds
        let cols = shape.dims()[1];
        for &col in &col_indices {
            if col >= cols {
                return Err(TorshError::InvalidArgument(format!(
                    "Column index {col} out of bounds for {cols} columns"
                )));
            }
        }

        Ok(Self {
            row_ptr,
            col_indices,
            values,
            shape,
            dtype: DType::F32,
            device: DeviceType::Cpu,
        })
    }

    /// Create from raw parts (used by custom kernels)
    pub fn from_raw_parts(
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f32>,
        shape: Shape,
    ) -> TorshResult<Self> {
        Self::new(row_ptr, col_indices, values, shape)
    }

    /// Create an empty CSR tensor with given shape
    pub fn empty(shape: Shape) -> TorshResult<Self> {
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "CSR format currently only supports 2D tensors".to_string(),
            ));
        }

        let rows = shape.dims()[0];
        let row_ptr = vec![0; rows + 1];
        let col_indices = Vec::new();
        let values = Vec::new();

        Ok(Self {
            row_ptr,
            col_indices,
            values,
            shape,
            dtype: DType::F32,
            device: DeviceType::Cpu,
        })
    }

    /// Create from dense tensor
    pub fn from_dense(dense: &Tensor, threshold: f32) -> TorshResult<Self> {
        let coo = CooTensor::from_dense(dense, threshold)?;
        Self::from_coo(&coo)
    }

    /// Create from COO tensor
    pub fn from_coo(coo: &CooTensor) -> TorshResult<Self> {
        let shape = coo.shape().clone();
        let rows = shape.dims()[0];

        // Get sorted triplets
        let mut coo_sorted = coo.clone();
        coo_sorted.sort_indices();
        let triplets = coo_sorted.triplets();

        // Build CSR format
        let mut row_ptr = vec![0];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let mut current_row = 0;

        for (row, col, val) in triplets {
            // Fill in empty rows
            while current_row < row {
                row_ptr.push(col_indices.len());
                current_row += 1;
            }

            col_indices.push(col);
            values.push(val);
        }

        // Fill in any remaining empty rows
        while current_row < rows {
            row_ptr.push(col_indices.len());
            current_row += 1;
        }

        Self::new(row_ptr, col_indices, values, shape)
    }

    /// Get values for a specific row
    pub fn get_row(&self, row: usize) -> TorshResult<(Vec<usize>, Vec<f32>)> {
        if row >= self.shape.dims()[0] {
            return Err(TorshError::InvalidArgument(format!(
                "Row index {row} out of bounds"
            )));
        }

        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        let cols = self.col_indices[start..end].to_vec();
        let vals = self.values[start..end].to_vec();

        Ok((cols, vals))
    }

    /// Get value at specific position (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row >= self.shape.dims()[0] || col >= self.shape.dims()[1] {
            return None;
        }

        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        // Binary search for the column in this row
        for i in start..end {
            if self.col_indices[i] == col {
                return Some(self.values[i]);
            }
            if self.col_indices[i] > col {
                break; // Column indices are sorted
            }
        }

        None
    }

    /// Get triplets (row, col, value) for all non-zero elements
    pub fn triplets(&self) -> Vec<(usize, usize, f32)> {
        let mut result = Vec::new();

        for row in 0..self.shape.dims()[0] {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                result.push((row, self.col_indices[i], self.values[i]));
            }
        }

        result
    }

    /// Matrix-vector multiplication
    pub fn matvec(&self, vector: &Tensor) -> TorshResult<Tensor> {
        if vector.shape().ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "Vector must be 1-dimensional".to_string(),
            ));
        }

        if vector.shape().dims()[0] != self.shape.dims()[1] {
            return Err(TorshError::InvalidArgument(format!(
                "Vector length {} doesn't match matrix columns {}",
                vector.shape().dims()[0],
                self.shape.dims()[1]
            )));
        }

        let result = zeros::<f32>(&[self.shape.dims()[0]])?;

        for row in 0..self.shape.dims()[0] {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            let mut sum = 0.0;
            for i in start..end {
                let col = self.col_indices[i];
                let val = self.values[i];
                sum += val * vector.get(&[col])?;
            }

            result.set(&[row], sum)?;
        }

        Ok(result)
    }

    /// Get row pointers
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Get column indices
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Get values
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Create CSR tensor from triplets (row_indices, col_indices, values)
    pub fn from_triplets(
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f32>,
        shape: [usize; 2],
    ) -> TorshResult<Self> {
        if row_indices.len() != col_indices.len() || row_indices.len() != values.len() {
            return Err(TorshError::InvalidArgument(
                "Row indices, column indices, and values must have the same length".to_string(),
            ));
        }

        let rows = shape[0];
        let cols = shape[1];

        // Build CSR format from triplets
        let mut row_ptr = vec![0; rows + 1];

        // Count non-zeros per row
        for &row in &row_indices {
            if row >= rows {
                return Err(TorshError::InvalidArgument(format!(
                    "Row index {} out of bounds for {} rows",
                    row, rows
                )));
            }
            row_ptr[row + 1] += 1;
        }

        // Convert counts to offsets
        for i in 1..=rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        // Sort entries by row, then by column
        let mut triplets: Vec<(usize, usize, f32)> = row_indices
            .into_iter()
            .zip(col_indices.into_iter())
            .zip(values.into_iter())
            .map(|((r, c), v)| (r, c, v))
            .collect();

        triplets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let sorted_col_indices: Vec<usize> = triplets.iter().map(|(_, c, _)| *c).collect();
        let sorted_values: Vec<f32> = triplets.iter().map(|(_, _, v)| *v).collect();

        let shape = Shape::new(vec![rows, cols]);
        Self::new(row_ptr, sorted_col_indices, sorted_values, shape)
    }

    /// Convert CSR tensor to dense tensor
    pub fn to_dense(&self) -> TorshResult<Tensor<f32>> {
        let rows = self.shape.dims()[0];
        let cols = self.shape.dims()[1];
        let mut dense_data = vec![0.0f32; rows * cols];

        for row in 0..rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for idx in start..end {
                let col = self.col_indices[idx];
                let val = self.values[idx];
                dense_data[row * cols + col] = val;
            }
        }

        Tensor::from_data(dense_data, vec![rows, cols], self.device)
    }

    /// Get data for a specific row
    pub fn get_row_data(&self, row: usize) -> TorshResult<(Vec<usize>, Vec<f32>)> {
        if row >= self.shape.dims()[0] {
            return Err(TorshError::IndexError {
                index: row,
                size: self.shape.dims()[0],
            });
        }

        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        let col_indices = self.col_indices[start..end].to_vec();
        let values = self.values[start..end].to_vec();

        Ok((col_indices, values))
    }

    /// Efficient matrix multiplication with dense tensor
    ///
    /// Implements sparse-dense matrix multiplication directly on CSR format
    /// without converting to dense, providing significant performance improvements.
    ///
    /// For CSR matrix A and dense matrix B, computes C = A × B where:
    /// - A is sparse (m × k) in CSR format
    /// - B is dense (k × n)
    /// - C is dense (m × n)
    ///
    /// Time complexity: O(nnz * n) where nnz is number of non-zeros in A
    /// Space complexity: O(m * n) for result matrix
    pub fn matmul(&self, other: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
        let other_shape = other.shape();

        // Validate dimensions for matrix multiplication
        if other_shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Dense tensor must be 2D for matrix multiplication".to_string(),
            ));
        }

        let [m, k] = [self.shape.dims()[0], self.shape.dims()[1]];
        let [k_other, n] = [other_shape.dims()[0], other_shape.dims()[1]];

        if k != k_other {
            return Err(TorshError::InvalidArgument(format!(
                "Incompatible dimensions for matrix multiplication: ({}, {}) × ({}, {})",
                m, k, k_other, n
            )));
        }

        // Create result tensor with zeros
        let result_shape = vec![m, n];
        let result = zeros::<f32>(&result_shape)?;

        // Get data access for efficient computation
        let other_data = other
            .data()
            .map_err(|e| TorshError::ComputeError(e.to_string()))?;
        let mut result_data = result
            .data()
            .map_err(|e| TorshError::ComputeError(e.to_string()))?;

        // Perform sparse-dense matrix multiplication
        // For each row in the sparse matrix
        for row in 0..m {
            let row_start = self.row_ptr[row];
            let row_end = self.row_ptr[row + 1];

            // For each non-zero element in this row
            for idx in row_start..row_end {
                let col = self.col_indices[idx];
                let sparse_val = self.values[idx];

                // Multiply sparse element with corresponding row of dense matrix
                // and accumulate into result row
                for result_col in 0..n {
                    let dense_val = other_data[col * n + result_col];
                    result_data[row * n + result_col] += sparse_val * dense_val;
                }
            }
        }

        Ok(result)
    }
}

impl SparseTensor for CsrTensor {
    fn format(&self) -> SparseFormat {
        SparseFormat::Csr
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> DeviceType {
        self.device
    }

    fn nnz(&self) -> usize {
        self.values.len()
    }

    fn to_dense(&self) -> TorshResult<Tensor> {
        let dense = zeros::<f32>(self.shape.dims())?;

        for row in 0..self.shape.dims()[0] {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                let col = self.col_indices[i];
                let val = self.values[i];
                dense.set(&[row, col], val)?;
            }
        }

        Ok(dense)
    }

    fn to_coo(&self) -> TorshResult<CooTensor> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..self.shape.dims()[0] {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                row_indices.push(row);
                col_indices.push(self.col_indices[i]);
                values.push(self.values[i]);
            }
        }

        CooTensor::new(row_indices, col_indices, values, self.shape.clone())
    }

    fn to_csr(&self) -> TorshResult<CsrTensor> {
        Ok(self.clone())
    }

    fn to_csc(&self) -> TorshResult<crate::CscTensor> {
        let coo = self.to_coo()?;
        crate::CscTensor::from_coo(&coo)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Clone for CsrTensor {
    fn clone(&self) -> Self {
        Self {
            row_ptr: self.row_ptr.clone(),
            col_indices: self.col_indices.clone(),
            values: self.values.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        }
    }
}

impl CsrTensor {
    /// Transpose the matrix
    pub fn transpose(&self) -> TorshResult<crate::CscTensor> {
        // CSR transpose is CSC, so convert to CSC
        self.to_csc()
    }

    /// Sum all elements in the matrix
    pub fn sum(&self) -> TorshResult<f32> {
        Ok(self.values.iter().sum())
    }

    /// Scale the matrix by a scalar
    pub fn scale(&self, scalar: f32) -> TorshResult<Self> {
        let scaled_values: Vec<f32> = self.values.iter().map(|&v| v * scalar).collect();
        CsrTensor::new(
            self.row_ptr.clone(),
            self.col_indices.clone(),
            scaled_values,
            self.shape.clone(),
        )
    }

    /// Compute the norm of the matrix
    pub fn norm(&self, p: f32) -> TorshResult<f32> {
        if p == 2.0 {
            // L2 norm (Frobenius norm for matrices)
            Ok(self.values.iter().map(|&v| v * v).sum::<f32>().sqrt())
        } else if p == 1.0 {
            // L1 norm
            Ok(self.values.iter().map(|&v| v.abs()).sum())
        } else {
            // General p-norm
            Ok(self
                .values
                .iter()
                .map(|&v| v.abs().powf(p))
                .sum::<f32>()
                .powf(1.0 / p))
        }
    }

    /// Extract diagonal elements
    pub fn diagonal(&self) -> TorshResult<Vec<f32>> {
        let min_dim = self.shape.dims()[0].min(self.shape.dims()[1]);
        let mut diag = vec![0.0; min_dim];

        #[allow(clippy::needless_range_loop)]
        for row in 0..min_dim {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                if self.col_indices[i] == row {
                    diag[row] = self.values[i];
                    break;
                }
            }
        }

        Ok(diag)
    }

    /// Add two sparse matrices
    pub fn add(&self, other: &CsrTensor) -> TorshResult<CsrTensor> {
        if self.shape.dims() != other.shape.dims() {
            return Err(TorshError::InvalidArgument(
                "Matrices must have the same shape for addition".to_string(),
            ));
        }

        // For simplicity, convert both to COO, add, then convert back to CSR
        let coo_self = self.to_coo()?;
        let coo_other = other.to_coo()?;

        // Combine triplets
        let mut triplets = Vec::new();

        // Add triplets from self
        for i in 0..coo_self.nnz() {
            triplets.push((
                coo_self.row_indices()[i],
                coo_self.col_indices()[i],
                coo_self.values()[i],
            ));
        }

        // Add triplets from other
        for i in 0..coo_other.nnz() {
            triplets.push((
                coo_other.row_indices()[i],
                coo_other.col_indices()[i],
                coo_other.values()[i],
            ));
        }

        // Create COO from combined triplets and convert to CSR
        let shape = (self.shape.dims()[0], self.shape.dims()[1]);
        let coo_result = crate::CooTensor::from_triplets(triplets, shape)?;
        CsrTensor::from_coo(&coo_result)
    }

    /// Compute the density (fraction of non-zeros)
    pub fn density(&self) -> f32 {
        let total_elements = self.shape().numel();
        if total_elements == 0 {
            0.0
        } else {
            self.nnz() as f32 / total_elements as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_csr_creation() {
        // Matrix:
        // [0, 1, 0]
        // [0, 0, 2]
        // [3, 0, 0]
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(vec![3, 3]);

        let csr = CsrTensor::new(row_ptr, col_indices, values, shape).unwrap();
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn test_csr_matvec() {
        let row_ptr = vec![0, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(vec![3, 3]);

        let csr = CsrTensor::new(row_ptr, col_indices, values, shape).unwrap();
        let vector = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();

        let result = csr.matvec(&vector).unwrap();

        // Expected: [0*1 + 1*2 + 0*3, 0*1 + 0*2 + 2*3, 3*1 + 0*2 + 0*3] = [2, 6, 3]
        assert_eq!(result.get(&[0]).unwrap(), 2.0);
        assert_eq!(result.get(&[1]).unwrap(), 6.0);
        assert_eq!(result.get(&[2]).unwrap(), 3.0);
    }
}
