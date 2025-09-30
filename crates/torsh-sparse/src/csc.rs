//! CSC (Compressed Sparse Column) sparse tensor format

use crate::{CooTensor, SparseFormat, SparseTensor, TorshResult};
use torsh_core::{device::DeviceType, DType, Shape, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

/// CSC (Compressed Sparse Column) format sparse tensor
pub struct CscTensor {
    /// Column pointers (size: cols + 1)
    col_ptr: Vec<usize>,
    /// Row indices
    row_indices: Vec<usize>,
    /// Non-zero values
    values: Vec<f32>,
    /// Shape of the tensor
    shape: Shape,
    /// Data type
    dtype: DType,
    /// Device
    device: DeviceType,
}

impl CscTensor {
    /// Create a new CSC tensor
    pub fn new(
        col_ptr: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f32>,
        shape: Shape,
    ) -> TorshResult<Self> {
        // Validate inputs
        if row_indices.len() != values.len() {
            return Err(TorshError::InvalidArgument(
                "Row indices and values must have the same length".to_string(),
            ));
        }

        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "CSC format currently only supports 2D tensors".to_string(),
            ));
        }

        let cols = shape.dims()[1];
        if col_ptr.len() != cols + 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Column pointer length must be cols + 1, got {} for {} columns",
                col_ptr.len(),
                cols
            )));
        }

        // Validate row indices are within bounds
        let rows = shape.dims()[0];
        for &row in &row_indices {
            if row >= rows {
                return Err(TorshError::InvalidArgument(format!(
                    "Row index {row} out of bounds for {rows} rows"
                )));
            }
        }

        Ok(Self {
            col_ptr,
            row_indices,
            values,
            shape,
            dtype: DType::F32,
            device: DeviceType::Cpu,
        })
    }

    /// Create from raw parts (used by custom kernels)
    pub fn from_raw_parts(
        col_ptr: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f32>,
        shape: Shape,
    ) -> TorshResult<Self> {
        Self::new(col_ptr, row_indices, values, shape)
    }

    /// Create an empty CSC tensor with given shape
    pub fn empty(shape: Shape) -> TorshResult<Self> {
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "CSC format currently only supports 2D tensors".to_string(),
            ));
        }

        let cols = shape.dims()[1];
        let col_ptr = vec![0; cols + 1];
        let row_indices = Vec::new();
        let values = Vec::new();

        Ok(Self {
            col_ptr,
            row_indices,
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
        let cols = shape.dims()[1];

        // Sort triplets by column then row
        let mut triplets = coo.triplets();
        triplets.sort_by_key(|&(row, col, _)| (col, row));

        // Build CSC format
        let mut col_ptr = vec![0];
        let mut row_indices = Vec::new();
        let mut values = Vec::new();

        let mut current_col = 0;

        for (row, col, val) in triplets {
            // Fill in empty columns
            while current_col < col {
                col_ptr.push(row_indices.len());
                current_col += 1;
            }

            row_indices.push(row);
            values.push(val);
        }

        // Fill in any remaining empty columns
        while current_col < cols {
            col_ptr.push(row_indices.len());
            current_col += 1;
        }

        Self::new(col_ptr, row_indices, values, shape)
    }

    /// Get values for a specific column
    pub fn get_col(&self, col: usize) -> TorshResult<(Vec<usize>, Vec<f32>)> {
        if col >= self.shape.dims()[1] {
            return Err(TorshError::InvalidArgument(format!(
                "Column index {col} out of bounds"
            )));
        }

        let start = self.col_ptr[col];
        let end = self.col_ptr[col + 1];

        let rows = self.row_indices[start..end].to_vec();
        let vals = self.values[start..end].to_vec();

        Ok((rows, vals))
    }

    /// Vector-matrix multiplication (vector @ matrix)
    pub fn vecmat(&self, vector: &Tensor) -> TorshResult<Tensor> {
        if vector.shape().ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "Vector must be 1-dimensional".to_string(),
            ));
        }

        if vector.shape().dims()[0] != self.shape.dims()[0] {
            return Err(TorshError::InvalidArgument(format!(
                "Vector length {} doesn't match matrix rows {}",
                vector.shape().dims()[0],
                self.shape.dims()[0]
            )));
        }

        let result = zeros::<f32>(&[self.shape.dims()[1]])?;

        for col in 0..self.shape.dims()[1] {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];

            let mut sum = 0.0;
            for i in start..end {
                let row = self.row_indices[i];
                let val = self.values[i];
                sum += vector.get(&[row])? * val;
            }

            result.set(&[col], sum)?;
        }

        Ok(result)
    }

    /// Get column pointers
    pub fn col_ptr(&self) -> &Vec<usize> {
        &self.col_ptr
    }

    /// Get row indices
    pub fn row_indices(&self) -> &Vec<usize> {
        &self.row_indices
    }

    /// Get values
    pub fn values(&self) -> &Vec<f32> {
        &self.values
    }

    /// Transpose to CSR (which is just swapping column and row operations)
    pub fn transpose_to_csr(&self) -> crate::CsrTensor {
        crate::CsrTensor::new(
            self.col_ptr.clone(),
            self.row_indices.clone(),
            self.values.clone(),
            Shape::new(vec![self.shape.dims()[1], self.shape.dims()[0]]),
        )
        .unwrap() // Safe because we know the format is valid
    }
}

impl SparseTensor for CscTensor {
    fn format(&self) -> SparseFormat {
        SparseFormat::Csc
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

        for col in 0..self.shape.dims()[1] {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];

            for i in start..end {
                let row = self.row_indices[i];
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

        for col in 0..self.shape.dims()[1] {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];

            for i in start..end {
                row_indices.push(self.row_indices[i]);
                col_indices.push(col);
                values.push(self.values[i]);
            }
        }

        CooTensor::new(row_indices, col_indices, values, self.shape.clone())
    }

    fn to_csr(&self) -> TorshResult<crate::CsrTensor> {
        let coo = self.to_coo()?;
        crate::CsrTensor::from_coo(&coo)
    }

    fn to_csc(&self) -> TorshResult<CscTensor> {
        Ok(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Clone for CscTensor {
    fn clone(&self) -> Self {
        Self {
            col_ptr: self.col_ptr.clone(),
            row_indices: self.row_indices.clone(),
            values: self.values.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_csc_creation() {
        // Matrix:
        // [0, 3, 0]
        // [1, 0, 0]
        // [0, 0, 2]
        let col_ptr = vec![0, 1, 2, 3];
        let row_indices = vec![1, 0, 2];
        let values = vec![1.0, 3.0, 2.0];
        let shape = Shape::new(vec![3, 3]);

        let csc = CscTensor::new(col_ptr, row_indices, values, shape).unwrap();
        assert_eq!(csc.nnz(), 3);
    }

    #[test]
    fn test_csc_vecmat() {
        let col_ptr = vec![0, 1, 2, 3];
        let row_indices = vec![1, 0, 2];
        let values = vec![1.0, 3.0, 2.0];
        let shape = Shape::new(vec![3, 3]);

        let csc = CscTensor::new(col_ptr, row_indices, values, shape).unwrap();
        let vector = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();

        let result = csc.vecmat(&vector).unwrap();

        // Expected: [1*0 + 2*1 + 3*0, 1*3 + 2*0 + 3*0, 1*0 + 2*0 + 3*2] = [2, 3, 6]
        assert_eq!(result.get(&[0]).unwrap(), 2.0);
        assert_eq!(result.get(&[1]).unwrap(), 3.0);
        assert_eq!(result.get(&[2]).unwrap(), 6.0);
    }

    #[test]
    fn test_csc_to_dense() {
        let col_ptr = vec![0, 1, 2, 3];
        let row_indices = vec![1, 0, 2];
        let values = vec![1.0, 3.0, 2.0];
        let shape = Shape::new(vec![3, 3]);

        let csc = CscTensor::new(col_ptr, row_indices, values, shape).unwrap();
        let dense = csc.to_dense().unwrap();

        assert_eq!(dense.get(&[0, 1]).unwrap(), 3.0);
        assert_eq!(dense.get(&[1, 0]).unwrap(), 1.0);
        assert_eq!(dense.get(&[2, 2]).unwrap(), 2.0);
    }
}
