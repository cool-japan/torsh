//! ELL (ELLPACK) sparse tensor format
//!
//! The ELL format is efficient for sparse matrices where all rows have
//! approximately the same number of non-zero elements.

use crate::{CooTensor, SparseFormat, SparseTensor, TorshResult};
use torsh_core::{device::DeviceType, DType, Shape, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

/// ELL (ELLPACK) format sparse tensor
pub struct EllTensor {
    /// Column indices matrix (rows x max_nnz_per_row)
    col_indices: Vec<usize>,
    /// Values matrix (rows x max_nnz_per_row)
    values: Vec<f32>,
    /// Shape of the tensor
    shape: Shape,
    /// Maximum number of non-zeros per row
    max_nnz_per_row: usize,
    /// Data type
    dtype: DType,
    /// Device
    device: DeviceType,
}

impl EllTensor {
    /// Create a new ELL tensor
    pub fn new(
        col_indices: Vec<usize>,
        values: Vec<f32>,
        shape: Shape,
        max_nnz_per_row: usize,
    ) -> TorshResult<Self> {
        // Validate inputs
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "ELL format currently only supports 2D tensors".to_string(),
            ));
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];
        let expected_size = rows * max_nnz_per_row;

        if col_indices.len() != expected_size {
            return Err(TorshError::InvalidArgument(format!(
                "Column indices size mismatch: expected {}, got {}",
                expected_size,
                col_indices.len()
            )));
        }

        if values.len() != expected_size {
            return Err(TorshError::InvalidArgument(format!(
                "Values size mismatch: expected {}, got {}",
                expected_size,
                values.len()
            )));
        }

        // Validate column indices are within bounds (allowing for padding)
        for &col in &col_indices {
            if col >= cols && col != usize::MAX {
                return Err(TorshError::InvalidArgument(format!(
                    "Column index {col} out of bounds for {cols} columns"
                )));
            }
        }

        Ok(Self {
            col_indices,
            values,
            shape,
            max_nnz_per_row,
            dtype: DType::F32,
            device: DeviceType::Cpu,
        })
    }

    /// Create from COO tensor
    pub fn from_coo(coo: &CooTensor) -> TorshResult<Self> {
        let shape = coo.shape().clone();
        let rows = shape.dims()[0];
        let triplets = coo.triplets();

        // Count non-zeros per row to determine max_nnz_per_row
        let mut row_nnz_counts = vec![0; rows];
        for (row, _, _) in &triplets {
            row_nnz_counts[*row] += 1;
        }

        let max_nnz_per_row = *row_nnz_counts.iter().max().unwrap_or(&0);
        if max_nnz_per_row == 0 {
            return Err(TorshError::InvalidArgument(
                "Cannot create ELL from empty matrix".to_string(),
            ));
        }

        // Group triplets by row
        let mut row_data: Vec<Vec<(usize, f32)>> = vec![Vec::new(); rows];
        for (row, col, val) in triplets {
            row_data[row].push((col, val));
        }

        // Sort each row by column index
        for row_triplets in &mut row_data {
            row_triplets.sort_by_key(|&(col, _)| col);
        }

        // Build ELL arrays
        let mut col_indices = Vec::with_capacity(rows * max_nnz_per_row);
        let mut values = Vec::with_capacity(rows * max_nnz_per_row);

        #[allow(clippy::needless_range_loop)]
        for row in 0..rows {
            let row_triplets = &row_data[row];

            // Add actual non-zeros
            for &(col, val) in row_triplets {
                col_indices.push(col);
                values.push(val);
            }

            // Pad with dummy values if necessary
            let padding_needed = max_nnz_per_row - row_triplets.len();
            for _ in 0..padding_needed {
                col_indices.push(usize::MAX); // Use MAX as invalid column index
                values.push(0.0);
            }
        }

        Self::new(col_indices, values, shape, max_nnz_per_row)
    }

    /// Create from dense tensor
    pub fn from_dense(dense: &Tensor, threshold: f32) -> TorshResult<Self> {
        let coo = CooTensor::from_dense(dense, threshold)?;
        Self::from_coo(&coo)
    }

    /// Get value at specific position
    pub fn get(&self, row: usize, col: usize) -> TorshResult<f32> {
        if row >= self.shape.dims()[0] || col >= self.shape.dims()[1] {
            return Err(TorshError::InvalidArgument(
                "Index out of bounds".to_string(),
            ));
        }

        let row_start = row * self.max_nnz_per_row;
        let row_end = row_start + self.max_nnz_per_row;

        for i in row_start..row_end {
            if self.col_indices[i] == col {
                return Ok(self.values[i]);
            } else if self.col_indices[i] == usize::MAX {
                // Reached padding, element not found
                break;
            }
        }

        Ok(0.0)
    }

    /// Get row data (column indices and values)
    pub fn get_row(&self, row: usize) -> TorshResult<(Vec<usize>, Vec<f32>)> {
        if row >= self.shape.dims()[0] {
            return Err(TorshError::InvalidArgument(format!(
                "Row index {row} out of bounds"
            )));
        }

        let row_start = row * self.max_nnz_per_row;
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for i in 0..self.max_nnz_per_row {
            let idx = row_start + i;
            let col = self.col_indices[idx];

            if col != usize::MAX {
                cols.push(col);
                vals.push(self.values[idx]);
            } else {
                break; // Reached padding
            }
        }

        Ok((cols, vals))
    }

    /// Matrix-vector multiplication optimized for ELL structure
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

        let rows = self.shape.dims()[0];
        let result = zeros::<f32>(&[rows])?;

        for row in 0..rows {
            let mut sum = 0.0;
            let row_start = row * self.max_nnz_per_row;

            for i in 0..self.max_nnz_per_row {
                let idx = row_start + i;
                let col = self.col_indices[idx];

                if col != usize::MAX {
                    sum += self.values[idx] * vector.get(&[col])?;
                } else {
                    break; // Reached padding
                }
            }

            result.set(&[row], sum)?;
        }

        Ok(result)
    }

    /// Get maximum number of non-zeros per row
    pub fn max_nnz_per_row(&self) -> usize {
        self.max_nnz_per_row
    }

    /// Calculate storage efficiency (ratio of actual non-zeros to allocated space)
    pub fn storage_efficiency(&self) -> f32 {
        let actual_nnz = self.nnz();
        let allocated_space = self.shape.dims()[0] * self.max_nnz_per_row;

        if allocated_space == 0 {
            0.0
        } else {
            actual_nnz as f32 / allocated_space as f32
        }
    }

    /// Get actual number of non-zeros per row
    pub fn nnz_per_row(&self) -> Vec<usize> {
        let rows = self.shape.dims()[0];
        let mut counts = Vec::with_capacity(rows);

        for row in 0..rows {
            let row_start = row * self.max_nnz_per_row;
            let mut count = 0;

            for i in 0..self.max_nnz_per_row {
                let idx = row_start + i;
                if self.col_indices[idx] != usize::MAX && self.values[idx].abs() > f32::EPSILON {
                    count += 1;
                } else if self.col_indices[idx] == usize::MAX {
                    break; // Reached padding
                }
            }

            counts.push(count);
        }

        counts
    }
}

impl SparseTensor for EllTensor {
    fn format(&self) -> SparseFormat {
        SparseFormat::Ell
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
        let mut count = 0;
        for i in 0..self.values.len() {
            if self.col_indices[i] != usize::MAX && self.values[i].abs() > f32::EPSILON {
                count += 1;
            }
        }
        count
    }

    fn to_dense(&self) -> TorshResult<Tensor> {
        let dense = zeros::<f32>(self.shape.dims())?;
        let rows = self.shape.dims()[0];

        for row in 0..rows {
            let row_start = row * self.max_nnz_per_row;

            for i in 0..self.max_nnz_per_row {
                let idx = row_start + i;
                let col = self.col_indices[idx];

                if col != usize::MAX {
                    let val = self.values[idx];
                    dense.set(&[row, col], val)?;
                } else {
                    break; // Reached padding
                }
            }
        }

        Ok(dense)
    }

    fn to_coo(&self) -> TorshResult<CooTensor> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let rows = self.shape.dims()[0];

        for row in 0..rows {
            let row_start = row * self.max_nnz_per_row;

            for i in 0..self.max_nnz_per_row {
                let idx = row_start + i;
                let col = self.col_indices[idx];

                if col != usize::MAX {
                    let val = self.values[idx];
                    if val.abs() > f32::EPSILON {
                        row_indices.push(row);
                        col_indices.push(col);
                        values.push(val);
                    }
                } else {
                    break; // Reached padding
                }
            }
        }

        CooTensor::new(row_indices, col_indices, values, self.shape.clone())
    }

    fn to_csr(&self) -> TorshResult<crate::CsrTensor> {
        let coo = self.to_coo()?;
        crate::CsrTensor::from_coo(&coo)
    }

    fn to_csc(&self) -> TorshResult<crate::CscTensor> {
        let coo = self.to_coo()?;
        crate::CscTensor::from_coo(&coo)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Clone for EllTensor {
    fn clone(&self) -> Self {
        Self {
            col_indices: self.col_indices.clone(),
            values: self.values.clone(),
            shape: self.shape.clone(),
            max_nnz_per_row: self.max_nnz_per_row,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ell_creation() {
        // Create a 3x3 matrix with max 2 non-zeros per row
        // Row 0: [1, 0, 2] -> cols=[0, 2], vals=[1, 2]
        // Row 1: [0, 3, 0] -> cols=[1], vals=[3]
        // Row 2: [4, 5, 0] -> cols=[0, 1], vals=[4, 5]
        let col_indices = vec![
            0,
            2, // Row 0: cols 0, 2
            1,
            usize::MAX, // Row 1: col 1, padding
            0,
            1, // Row 2: cols 0, 1
        ];
        let values = vec![
            1.0, 2.0, // Row 0: vals 1, 2
            3.0, 0.0, // Row 1: val 3, padding
            4.0, 5.0, // Row 2: vals 4, 5
        ];

        let shape = Shape::new(vec![3, 3]);
        let max_nnz_per_row = 2;

        let ell = EllTensor::new(col_indices, values, shape, max_nnz_per_row).unwrap();
        assert_eq!(ell.max_nnz_per_row(), 2);
    }

    #[test]
    fn test_ell_get() {
        let col_indices = vec![
            0,
            2, // Row 0
            1,
            usize::MAX, // Row 1
            0,
            1, // Row 2
        ];
        let values = vec![
            1.0, 2.0, // Row 0
            3.0, 0.0, // Row 1
            4.0, 5.0, // Row 2
        ];

        let shape = Shape::new(vec![3, 3]);
        let ell = EllTensor::new(col_indices, values, shape, 2).unwrap();

        assert_relative_eq!(ell.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(ell.get(0, 2).unwrap(), 2.0);
        assert_relative_eq!(ell.get(1, 1).unwrap(), 3.0);
        assert_relative_eq!(ell.get(2, 0).unwrap(), 4.0);
        assert_relative_eq!(ell.get(2, 1).unwrap(), 5.0);
        assert_relative_eq!(ell.get(0, 1).unwrap(), 0.0); // Zero element
    }

    #[test]
    fn test_ell_get_row() {
        let col_indices = vec![
            0,
            2, // Row 0
            1,
            usize::MAX, // Row 1
            0,
            1, // Row 2
        ];
        let values = vec![
            1.0, 2.0, // Row 0
            3.0, 0.0, // Row 1
            4.0, 5.0, // Row 2
        ];

        let shape = Shape::new(vec![3, 3]);
        let ell = EllTensor::new(col_indices, values, shape, 2).unwrap();

        let (cols, vals) = ell.get_row(0).unwrap();
        assert_eq!(cols, vec![0, 2]);
        assert_eq!(vals, vec![1.0, 2.0]);

        let (cols, vals) = ell.get_row(1).unwrap();
        assert_eq!(cols, vec![1]);
        assert_eq!(vals, vec![3.0]);
    }

    #[test]
    fn test_ell_to_dense() {
        let col_indices = vec![
            0,
            usize::MAX, // Row 0: only col 0
            1,
            usize::MAX, // Row 1: only col 1
        ];
        let values = vec![
            1.0, 0.0, // Row 0
            2.0, 0.0, // Row 1
        ];

        let shape = Shape::new(vec![2, 2]);
        let ell = EllTensor::new(col_indices, values, shape, 2).unwrap();
        let dense = ell.to_dense().unwrap();

        // Expected matrix:
        // [1, 0]
        // [0, 2]
        assert_relative_eq!(dense.get(&[0, 0]).unwrap(), 1.0);
        assert_relative_eq!(dense.get(&[0, 1]).unwrap(), 0.0);
        assert_relative_eq!(dense.get(&[1, 0]).unwrap(), 0.0);
        assert_relative_eq!(dense.get(&[1, 1]).unwrap(), 2.0);
    }

    #[test]
    fn test_ell_storage_efficiency() {
        let col_indices = vec![
            0,
            usize::MAX, // Row 0: 1 element, 1 padding
            1,
            usize::MAX, // Row 1: 1 element, 1 padding
        ];
        let values = vec![
            1.0, 0.0, // Row 0
            2.0, 0.0, // Row 1
        ];

        let shape = Shape::new(vec![2, 2]);
        let ell = EllTensor::new(col_indices, values, shape, 2).unwrap();

        // 2 actual non-zeros out of 4 allocated slots = 50% efficiency
        assert_relative_eq!(ell.storage_efficiency(), 0.5);

        let nnz_counts = ell.nnz_per_row();
        assert_eq!(nnz_counts, vec![1, 1]);
    }
}
