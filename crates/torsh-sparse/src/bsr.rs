//! BSR (Block Sparse Row) sparse tensor format
//!
//! The BSR format is efficient for sparse matrices that have dense blocks.
//! It stores the matrix as a collection of dense blocks.

use crate::{CooTensor, SparseFormat, SparseTensor, TorshResult};
use torsh_core::{device::DeviceType, DType, Shape, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

/// BSR (Block Sparse Row) format sparse tensor
pub struct BsrTensor {
    /// Block row pointers (size: block_rows + 1)
    block_row_ptr: Vec<usize>,
    /// Block column indices
    block_col_indices: Vec<usize>,
    /// Dense blocks stored in row-major order
    blocks: Vec<f32>,
    /// Shape of the tensor
    shape: Shape,
    /// Block size (rows, cols)
    block_size: (usize, usize),
    /// Data type
    dtype: DType,
    /// Device
    device: DeviceType,
}

impl BsrTensor {
    /// Create a new BSR tensor
    pub fn new(
        block_row_ptr: Vec<usize>,
        block_col_indices: Vec<usize>,
        blocks: Vec<f32>,
        shape: Shape,
        block_size: (usize, usize),
    ) -> TorshResult<Self> {
        // Validate inputs
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "BSR format currently only supports 2D tensors".to_string(),
            ));
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];
        let (block_rows, block_cols) = block_size;

        if rows % block_rows != 0 || cols % block_cols != 0 {
            return Err(TorshError::InvalidArgument(
                "Matrix dimensions must be divisible by block size".to_string(),
            ));
        }

        let num_block_rows = rows / block_rows;
        if block_row_ptr.len() != num_block_rows + 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Block row pointer length must be num_block_rows + 1, got {} for {} block rows",
                block_row_ptr.len(),
                num_block_rows
            )));
        }

        let num_blocks = block_col_indices.len();
        let expected_block_data_size = num_blocks * block_rows * block_cols;
        if blocks.len() != expected_block_data_size {
            return Err(TorshError::InvalidArgument(format!(
                "Block data size mismatch: expected {}, got {}",
                expected_block_data_size,
                blocks.len()
            )));
        }

        // Validate block column indices
        let num_block_cols = cols / block_cols;
        for &block_col in &block_col_indices {
            if block_col >= num_block_cols {
                return Err(TorshError::InvalidArgument(format!(
                    "Block column index {block_col} out of bounds for {num_block_cols} block columns"
                )));
            }
        }

        Ok(Self {
            block_row_ptr,
            block_col_indices,
            blocks,
            shape,
            block_size,
            dtype: DType::F32,
            device: DeviceType::Cpu,
        })
    }

    /// Create from COO tensor
    pub fn from_coo(coo: &CooTensor, block_size: (usize, usize)) -> TorshResult<Self> {
        let shape = coo.shape().clone();
        let (block_rows, block_cols) = block_size;
        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        if rows % block_rows != 0 || cols % block_cols != 0 {
            return Err(TorshError::InvalidArgument(
                "Matrix dimensions must be divisible by block size".to_string(),
            ));
        }

        let num_block_rows = rows / block_rows;
        let num_block_cols = cols / block_cols;

        // Convert to block coordinates
        let triplets = coo.triplets();
        let mut block_map = std::collections::HashMap::new();

        for (row, col, val) in triplets {
            let block_row = row / block_rows;
            let block_col = col / block_cols;
            let in_block_row = row % block_rows;
            let in_block_col = col % block_cols;

            let block_key = (block_row, block_col);
            let block_entry = block_map
                .entry(block_key)
                .or_insert_with(|| vec![0.0; block_rows * block_cols]);

            let block_index = in_block_row * block_cols + in_block_col;
            block_entry[block_index] = val;
        }

        // Convert to BSR format
        let mut block_row_ptr = vec![0];
        let mut block_col_indices = Vec::new();
        let mut blocks = Vec::new();

        for block_row in 0..num_block_rows {
            let mut row_blocks = Vec::new();

            for block_col in 0..num_block_cols {
                if let Some(block_data) = block_map.get(&(block_row, block_col)) {
                    row_blocks.push((block_col, block_data.clone()));
                }
            }

            // Sort by block column index
            row_blocks.sort_by_key(|&(block_col, _)| block_col);

            // Add to BSR data structures
            for (block_col, block_data) in row_blocks {
                block_col_indices.push(block_col);
                blocks.extend(block_data);
            }

            block_row_ptr.push(block_col_indices.len());
        }

        Self::new(block_row_ptr, block_col_indices, blocks, shape, block_size)
    }

    /// Get a specific block
    pub fn get_block(&self, block_row: usize, block_col: usize) -> TorshResult<Option<Vec<f32>>> {
        let num_block_rows = self.shape.dims()[0] / self.block_size.0;
        if block_row >= num_block_rows {
            return Err(TorshError::InvalidArgument(format!(
                "Block row index {block_row} out of bounds"
            )));
        }

        let start = self.block_row_ptr[block_row];
        let end = self.block_row_ptr[block_row + 1];

        // Search for the block column
        for i in start..end {
            if self.block_col_indices[i] == block_col {
                let block_size = self.block_size.0 * self.block_size.1;
                let block_start = i * block_size;
                let block_end = block_start + block_size;
                return Ok(Some(self.blocks[block_start..block_end].to_vec()));
            }
        }

        Ok(None)
    }

    /// Block matrix-vector multiplication
    pub fn block_matvec(&self, vector: &Tensor) -> TorshResult<Tensor> {
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
        let (block_rows, block_cols) = self.block_size;
        let num_block_rows = self.shape.dims()[0] / block_rows;

        for block_row in 0..num_block_rows {
            let start = self.block_row_ptr[block_row];
            let end = self.block_row_ptr[block_row + 1];

            for i in start..end {
                let block_col = self.block_col_indices[i];
                let block_size = block_rows * block_cols;
                let block_start = i * block_size;

                // Perform dense matrix-vector multiplication for this block
                for r in 0..block_rows {
                    let mut sum = 0.0;
                    for c in 0..block_cols {
                        let block_idx = block_start + r * block_cols + c;
                        let vector_idx = block_col * block_cols + c;
                        sum += self.blocks[block_idx] * vector.get(&[vector_idx])?;
                    }
                    let result_idx = block_row * block_rows + r;
                    let current_value = result.get(&[result_idx])?;
                    result.set(&[result_idx], current_value + sum)?;
                }
            }
        }

        Ok(result)
    }

    /// Get block size
    pub fn block_size(&self) -> (usize, usize) {
        self.block_size
    }

    /// Get number of blocks
    pub fn num_blocks(&self) -> usize {
        self.block_col_indices.len()
    }
}

impl SparseTensor for BsrTensor {
    fn format(&self) -> SparseFormat {
        SparseFormat::Bsr
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
        self.blocks
            .iter()
            .filter(|&&x| x.abs() > f32::EPSILON)
            .count()
    }

    fn to_dense(&self) -> TorshResult<Tensor> {
        let dense = zeros::<f32>(self.shape.dims())?;
        let (block_rows, block_cols) = self.block_size;
        let num_block_rows = self.shape.dims()[0] / block_rows;

        for block_row in 0..num_block_rows {
            let start = self.block_row_ptr[block_row];
            let end = self.block_row_ptr[block_row + 1];

            for i in start..end {
                let block_col = self.block_col_indices[i];
                let block_size = block_rows * block_cols;
                let block_start = i * block_size;

                // Copy block data to dense matrix
                for r in 0..block_rows {
                    for c in 0..block_cols {
                        let block_idx = block_start + r * block_cols + c;
                        let row = block_row * block_rows + r;
                        let col = block_col * block_cols + c;
                        dense.set(&[row, col], self.blocks[block_idx])?;
                    }
                }
            }
        }

        Ok(dense)
    }

    fn to_coo(&self) -> TorshResult<CooTensor> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let (block_rows, block_cols) = self.block_size;
        let num_block_rows = self.shape.dims()[0] / block_rows;

        for block_row in 0..num_block_rows {
            let start = self.block_row_ptr[block_row];
            let end = self.block_row_ptr[block_row + 1];

            for i in start..end {
                let block_col = self.block_col_indices[i];
                let block_size = block_rows * block_cols;
                let block_start = i * block_size;

                // Extract non-zero elements from the block
                for r in 0..block_rows {
                    for c in 0..block_cols {
                        let block_idx = block_start + r * block_cols + c;
                        let val = self.blocks[block_idx];

                        if val.abs() > f32::EPSILON {
                            let row = block_row * block_rows + r;
                            let col = block_col * block_cols + c;
                            row_indices.push(row);
                            col_indices.push(col);
                            values.push(val);
                        }
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

impl Clone for BsrTensor {
    fn clone(&self) -> Self {
        Self {
            block_row_ptr: self.block_row_ptr.clone(),
            block_col_indices: self.block_col_indices.clone(),
            blocks: self.blocks.clone(),
            shape: self.shape.clone(),
            block_size: self.block_size,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bsr_creation() {
        // Create a 4x4 matrix with 2x2 blocks
        // Block structure:
        // [A, B]
        // [C, D]
        // where each block is 2x2

        let block_row_ptr = vec![0, 2, 3]; // 2 block rows
        let block_col_indices = vec![0, 1, 0]; // First row has blocks 0,1; second row has block 0
        let blocks = vec![
            // Block (0,0)
            1.0, 2.0, 3.0, 4.0, // Block (0,1)
            5.0, 6.0, 7.0, 8.0, // Block (1,0)
            9.0, 10.0, 11.0, 12.0,
        ];

        let shape = Shape::new(vec![4, 4]);
        let block_size = (2, 2);

        let bsr =
            BsrTensor::new(block_row_ptr, block_col_indices, blocks, shape, block_size).unwrap();
        assert_eq!(bsr.num_blocks(), 3);
        assert_eq!(bsr.block_size(), (2, 2));
    }

    #[test]
    fn test_bsr_to_dense() {
        // Simple 2x2 BSR with one 2x2 block
        let block_row_ptr = vec![0, 1];
        let block_col_indices = vec![0];
        let blocks = vec![1.0, 2.0, 3.0, 4.0];

        let shape = Shape::new(vec![2, 2]);
        let block_size = (2, 2);

        let bsr =
            BsrTensor::new(block_row_ptr, block_col_indices, blocks, shape, block_size).unwrap();
        let dense = bsr.to_dense().unwrap();

        assert_eq!(dense.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(dense.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(dense.get(&[1, 0]).unwrap(), 3.0);
        assert_eq!(dense.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_bsr_get_block() {
        let block_row_ptr = vec![0, 1];
        let block_col_indices = vec![0];
        let blocks = vec![1.0, 2.0, 3.0, 4.0];

        let shape = Shape::new(vec![2, 2]);
        let block_size = (2, 2);

        let bsr =
            BsrTensor::new(block_row_ptr, block_col_indices, blocks, shape, block_size).unwrap();

        let block = bsr.get_block(0, 0).unwrap().unwrap();
        assert_eq!(block, vec![1.0, 2.0, 3.0, 4.0]);

        let no_block = bsr.get_block(0, 1).unwrap();
        assert!(no_block.is_none());
    }
}
