//! DIA (Diagonal) sparse tensor format
//!
//! The DIA format is efficient for sparse matrices with a diagonal structure.
//! It stores the matrix as a collection of diagonals.

use crate::{CooTensor, SparseFormat, SparseTensor, TorshResult};
use torsh_core::{device::DeviceType, DType, Shape, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

/// DIA (Diagonal) format sparse tensor
pub struct DiaTensor {
    /// Diagonal data stored as columns
    /// Each column represents one diagonal
    data: Vec<f32>,
    /// Diagonal offsets (negative for below main diagonal, positive for above)
    offsets: Vec<isize>,
    /// Shape of the tensor
    shape: Shape,
    /// Number of diagonals
    num_diagonals: usize,
    /// Data type
    dtype: DType,
    /// Device
    device: DeviceType,
}

impl DiaTensor {
    /// Create a new DIA tensor
    pub fn new(data: Vec<f32>, offsets: Vec<isize>, shape: Shape) -> TorshResult<Self> {
        // Validate inputs
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "DIA format currently only supports 2D tensors".to_string(),
            ));
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];
        let num_diagonals = offsets.len();

        if data.len() != num_diagonals * rows.max(cols) {
            return Err(TorshError::InvalidArgument(format!(
                "Data size mismatch: expected {}, got {}",
                num_diagonals * rows.max(cols),
                data.len()
            )));
        }

        // Validate offsets are within bounds
        let max_offset = (cols as isize) - 1;
        let min_offset = -((rows as isize) - 1);

        for &offset in &offsets {
            if offset > max_offset || offset < min_offset {
                return Err(TorshError::InvalidArgument(format!(
                    "Offset {offset} out of bounds [{min_offset}, {max_offset}]"
                )));
            }
        }

        Ok(Self {
            data,
            offsets,
            shape,
            num_diagonals,
            dtype: DType::F32,
            device: DeviceType::Cpu,
        })
    }

    /// Create from COO tensor
    pub fn from_coo(coo: &CooTensor) -> TorshResult<Self> {
        let shape = coo.shape().clone();
        let rows = shape.dims()[0];
        let cols = shape.dims()[1];
        let triplets = coo.triplets();

        // Collect all unique diagonal offsets
        let mut offset_set = std::collections::BTreeSet::new();
        for (row, col, _) in &triplets {
            let offset = (*col as isize) - (*row as isize);
            offset_set.insert(offset);
        }

        let offsets: Vec<isize> = offset_set.into_iter().collect();
        let num_diagonals = offsets.len();
        let max_dim = rows.max(cols);

        // Initialize data matrix
        let mut data = vec![0.0; num_diagonals * max_dim];

        // Fill diagonal data
        for (row, col, val) in triplets {
            let offset = (col as isize) - (row as isize);

            if let Some(diag_idx) = offsets.iter().position(|&o| o == offset) {
                let data_idx = diag_idx * max_dim + row;
                data[data_idx] = val;
            }
        }

        Self::new(data, offsets, shape)
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

        let offset = (col as isize) - (row as isize);

        if let Some(diag_idx) = self.offsets.iter().position(|&o| o == offset) {
            let max_dim = self.shape.dims()[0].max(self.shape.dims()[1]);
            let data_idx = diag_idx * max_dim + row;
            Ok(self.data[data_idx])
        } else {
            Ok(0.0)
        }
    }

    /// Set value at specific position
    pub fn set(&mut self, row: usize, col: usize, value: f32) -> TorshResult<()> {
        if row >= self.shape.dims()[0] || col >= self.shape.dims()[1] {
            return Err(TorshError::InvalidArgument(
                "Index out of bounds".to_string(),
            ));
        }

        let offset = (col as isize) - (row as isize);

        if let Some(diag_idx) = self.offsets.iter().position(|&o| o == offset) {
            let max_dim = self.shape.dims()[0].max(self.shape.dims()[1]);
            let data_idx = diag_idx * max_dim + row;
            self.data[data_idx] = value;
            Ok(())
        } else {
            // Need to add new diagonal
            if value.abs() > f32::EPSILON {
                return Err(TorshError::InvalidArgument(
                    "Cannot add new diagonal to existing DIA tensor".to_string(),
                ));
            }
            Ok(())
        }
    }

    /// Matrix-vector multiplication optimized for diagonal structure
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
        let cols = self.shape.dims()[1];
        let max_dim = rows.max(cols);
        let result = zeros::<f32>(&[rows])?;

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let start_row = if offset >= 0 { 0 } else { (-offset) as usize };
            let start_col = if offset >= 0 { offset as usize } else { 0 };

            let diag_length = if offset >= 0 {
                (rows).min(cols - (offset as usize))
            } else {
                (rows - ((-offset) as usize)).min(cols)
            };

            for i in 0..diag_length {
                let row = start_row + i;
                let col = start_col + i;

                if row < rows && col < cols {
                    let data_idx = diag_idx * max_dim + row;
                    let val = self.data[data_idx];

                    if val.abs() > f32::EPSILON {
                        let current = result.get(&[row])?;
                        result.set(&[row], current + val * vector.get(&[col])?)?;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Get diagonal offsets
    pub fn offsets(&self) -> &[isize] {
        &self.offsets
    }

    /// Get number of diagonals
    pub fn num_diagonals(&self) -> usize {
        self.num_diagonals
    }

    /// Extract a specific diagonal
    pub fn get_diagonal(&self, offset: isize) -> TorshResult<Vec<f32>> {
        if let Some(diag_idx) = self.offsets.iter().position(|&o| o == offset) {
            let rows = self.shape.dims()[0];
            let cols = self.shape.dims()[1];
            let max_dim = rows.max(cols);

            let start_row = if offset >= 0 { 0 } else { (-offset) as usize };

            let diag_length = if offset >= 0 {
                rows.min(cols - (offset as usize))
            } else {
                (rows - ((-offset) as usize)).min(cols)
            };

            let mut diagonal = Vec::with_capacity(diag_length);
            for i in 0..diag_length {
                let data_idx = diag_idx * max_dim + start_row + i;
                diagonal.push(self.data[data_idx]);
            }

            Ok(diagonal)
        } else {
            Err(TorshError::InvalidArgument(format!(
                "Diagonal with offset {offset} not found"
            )))
        }
    }
}

impl SparseTensor for DiaTensor {
    fn format(&self) -> SparseFormat {
        SparseFormat::Dia
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
        self.data
            .iter()
            .filter(|&&x| x.abs() > f32::EPSILON)
            .count()
    }

    fn to_dense(&self) -> TorshResult<Tensor> {
        let dense = zeros::<f32>(self.shape.dims())?;
        let rows = self.shape.dims()[0];
        let cols = self.shape.dims()[1];
        let max_dim = rows.max(cols);

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let start_row = if offset >= 0 { 0 } else { (-offset) as usize };
            let start_col = if offset >= 0 { offset as usize } else { 0 };

            let diag_length = if offset >= 0 {
                rows.min(cols - (offset as usize))
            } else {
                (rows - ((-offset) as usize)).min(cols)
            };

            for i in 0..diag_length {
                let row = start_row + i;
                let col = start_col + i;

                if row < rows && col < cols {
                    let data_idx = diag_idx * max_dim + row;
                    let val = self.data[data_idx];
                    dense.set(&[row, col], val)?;
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
        let cols = self.shape.dims()[1];
        let max_dim = rows.max(cols);

        for (diag_idx, &offset) in self.offsets.iter().enumerate() {
            let start_row = if offset >= 0 { 0 } else { (-offset) as usize };
            let start_col = if offset >= 0 { offset as usize } else { 0 };

            let diag_length = if offset >= 0 {
                rows.min(cols - (offset as usize))
            } else {
                (rows - ((-offset) as usize)).min(cols)
            };

            for i in 0..diag_length {
                let row = start_row + i;
                let col = start_col + i;

                if row < rows && col < cols {
                    let data_idx = diag_idx * max_dim + row;
                    let val = self.data[data_idx];

                    if val.abs() > f32::EPSILON {
                        row_indices.push(row);
                        col_indices.push(col);
                        values.push(val);
                    }
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

impl Clone for DiaTensor {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            offsets: self.offsets.clone(),
            shape: self.shape.clone(),
            num_diagonals: self.num_diagonals,
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
    fn test_dia_creation() {
        // Create a 3x3 tridiagonal matrix
        // Diagonals: -1 (below main), 0 (main), 1 (above main)
        let data = vec![
            // Diagonal -1 (below main): [0, 3, 6]
            0.0, 3.0, 6.0, // Diagonal 0 (main): [1, 4, 7]
            1.0, 4.0, 7.0, // Diagonal 1 (above main): [2, 5, 0]
            2.0, 5.0, 0.0,
        ];
        let offsets = vec![-1, 0, 1];
        let shape = Shape::new(vec![3, 3]);

        let dia = DiaTensor::new(data, offsets, shape).unwrap();
        assert_eq!(dia.num_diagonals(), 3);
        assert_eq!(dia.offsets(), &[-1, 0, 1]);
    }

    #[test]
    fn test_dia_get_set() {
        let data = vec![
            0.0, 3.0, 6.0, // diagonal -1
            1.0, 4.0, 7.0, // diagonal 0
            2.0, 5.0, 0.0, // diagonal 1
        ];
        let offsets = vec![-1, 0, 1];
        let shape = Shape::new(vec![3, 3]);

        let mut dia = DiaTensor::new(data, offsets, shape).unwrap();

        // Test get
        assert_relative_eq!(dia.get(0, 0).unwrap(), 1.0); // main diagonal
        assert_relative_eq!(dia.get(1, 0).unwrap(), 3.0); // below main diagonal
        assert_relative_eq!(dia.get(0, 1).unwrap(), 2.0); // above main diagonal
        assert_relative_eq!(dia.get(1, 2).unwrap(), 5.0); // above main diagonal element

        // Test set
        dia.set(1, 1, 10.0).unwrap();
        assert_relative_eq!(dia.get(1, 1).unwrap(), 10.0);
    }

    #[test]
    fn test_dia_to_dense() {
        let data = vec![
            0.0, 2.0, 0.0, // diagonal -1
            1.0, 3.0, 5.0, // diagonal 0
        ];
        let offsets = vec![-1, 0];
        let shape = Shape::new(vec![3, 3]);

        let dia = DiaTensor::new(data, offsets, shape).unwrap();
        let dense = dia.to_dense().unwrap();

        // Expected matrix:
        // [1, 0, 0]
        // [2, 3, 0]
        // [0, 0, 5]
        assert_relative_eq!(dense.get(&[0, 0]).unwrap(), 1.0);
        assert_relative_eq!(dense.get(&[1, 0]).unwrap(), 2.0);
        assert_relative_eq!(dense.get(&[1, 1]).unwrap(), 3.0);
        assert_relative_eq!(dense.get(&[2, 2]).unwrap(), 5.0);
        assert_relative_eq!(dense.get(&[0, 1]).unwrap(), 0.0);
    }

    #[test]
    fn test_dia_get_diagonal() {
        let data = vec![
            0.0, 2.0, 0.0, // diagonal -1
            1.0, 3.0, 5.0, // diagonal 0
        ];
        let offsets = vec![-1, 0];
        let shape = Shape::new(vec![3, 3]);

        let dia = DiaTensor::new(data, offsets, shape).unwrap();

        let main_diag = dia.get_diagonal(0).unwrap();
        assert_eq!(main_diag, vec![1.0, 3.0, 5.0]);

        let below_diag = dia.get_diagonal(-1).unwrap();
        assert_eq!(below_diag, vec![2.0, 0.0]); // Below main diagonal has 2 elements
    }
}
