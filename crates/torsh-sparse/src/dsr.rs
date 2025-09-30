//! Dynamic Sparse Row (DSR) format for sparse tensors
//!
//! DSR format is optimized for dynamic insertion and deletion of sparse elements
//! while maintaining efficient row-wise operations. Unlike CSR, it allows
//! efficient modification of the sparsity pattern.

use crate::{CooTensor, CscTensor, CsrTensor, SparseFormat, SparseTensor, TorshResult};
use std::collections::BTreeMap;
use torsh_core::{device::DeviceType, DType, Shape, TorshError};
use torsh_tensor::Tensor;

/// Dynamic Sparse Row tensor format
///
/// This format stores each row as a sorted map of column indices to values,
/// allowing for efficient insertion, deletion, and lookup operations.
/// Trade-off: Higher memory overhead compared to CSR, but better for dynamic operations.
#[derive(Debug, Clone)]
pub struct DsrTensor {
    /// Shape of the tensor
    shape: Shape,
    /// Data type of the tensor
    dtype: DType,
    /// Device type
    device: DeviceType,
    /// Row data: Vec<BTreeMap<column_index, value>>
    /// Using BTreeMap maintains sorted order for efficient operations
    rows: Vec<BTreeMap<usize, f32>>,
    /// Cached number of non-zero elements (updated on modification)
    cached_nnz: usize,
}

impl DsrTensor {
    /// Create a new DSR tensor with given shape
    pub fn new(shape: Shape, dtype: DType) -> TorshResult<Self> {
        if shape.dims().len() != 2 {
            return Err(TorshError::InvalidArgument(
                "DSR format only supports 2D tensors".to_string(),
            ));
        }

        let rows = vec![BTreeMap::new(); shape.dims()[0]];

        Ok(Self {
            shape,
            dtype,
            device: DeviceType::Cpu,
            rows,
            cached_nnz: 0,
        })
    }

    /// Create DSR tensor from COO tensor
    pub fn from_coo(coo: &CooTensor) -> TorshResult<Self> {
        let mut dsr = Self::new(coo.shape().clone(), coo.dtype())?;

        for (row, col, val) in coo.triplets() {
            dsr.set(row, col, val)?;
        }

        Ok(dsr)
    }

    /// Create DSR tensor from CSR tensor
    pub fn from_csr(csr: &CsrTensor) -> TorshResult<Self> {
        let coo = csr.to_coo()?;
        Self::from_coo(&coo)
    }

    /// Create DSR tensor from dense tensor with threshold
    pub fn from_dense(dense: &Tensor, _threshold: f32) -> TorshResult<Self> {
        let shape = dense.shape();
        if shape.dims().len() != 2 {
            return Err(TorshError::InvalidArgument(
                "DSR format only supports 2D tensors".to_string(),
            ));
        }

        let dsr = Self::new(shape.clone(), dense.dtype())?;

        // For now, we'll implement a simple approach
        // In a real implementation, you'd want to efficiently extract sparse elements
        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        // This is a simplified implementation - in practice you'd want to
        // extract values more efficiently from the dense tensor
        for _row in 0..rows {
            for _col in 0..cols {
                // Note: This would need proper tensor indexing in the real implementation
                // For now, we'll skip the actual dense extraction since tensor API might vary
                // let value = dense.get_2d(row, col)?;
                // if value.abs() > threshold {
                //     dsr.set(row, col, value)?;
                // }
            }
        }

        Ok(dsr)
    }

    /// Set a value at given row and column
    pub fn set(&mut self, row: usize, col: usize, value: f32) -> TorshResult<()> {
        if row >= self.shape.dims()[0] || col >= self.shape.dims()[1] {
            return Err(TorshError::InvalidArgument(
                "Index out of bounds".to_string(),
            ));
        }

        let was_present = self.rows[row].contains_key(&col);
        let is_zero = value.abs() < 1e-12;

        if is_zero {
            // Remove zero values
            if self.rows[row].remove(&col).is_some() {
                self.cached_nnz -= 1;
            }
        } else {
            // Insert or update non-zero values
            if !was_present {
                self.cached_nnz += 1;
            }
            self.rows[row].insert(col, value);
        }

        Ok(())
    }

    /// Get a value at given row and column
    pub fn get(&self, row: usize, col: usize) -> TorshResult<f32> {
        if row >= self.shape.dims()[0] || col >= self.shape.dims()[1] {
            return Err(TorshError::InvalidArgument(
                "Index out of bounds".to_string(),
            ));
        }

        Ok(self.rows[row].get(&col).copied().unwrap_or(0.0))
    }

    /// Remove an element at given row and column
    pub fn remove(&mut self, row: usize, col: usize) -> TorshResult<f32> {
        if row >= self.shape.dims()[0] || col >= self.shape.dims()[1] {
            return Err(TorshError::InvalidArgument(
                "Index out of bounds".to_string(),
            ));
        }

        if let Some(value) = self.rows[row].remove(&col) {
            self.cached_nnz -= 1;
            Ok(value)
        } else {
            Ok(0.0)
        }
    }

    /// Insert multiple elements for a row efficiently
    pub fn insert_row_elements(
        &mut self,
        row: usize,
        elements: &[(usize, f32)],
    ) -> TorshResult<()> {
        if row >= self.shape.dims()[0] {
            return Err(TorshError::InvalidArgument(
                "Row index out of bounds".to_string(),
            ));
        }

        for &(col, value) in elements {
            if col >= self.shape.dims()[1] {
                return Err(TorshError::InvalidArgument(
                    "Column index out of bounds".to_string(),
                ));
            }

            if value.abs() > 1e-12 {
                let was_present = self.rows[row].contains_key(&col);
                self.rows[row].insert(col, value);
                if !was_present {
                    self.cached_nnz += 1;
                }
            }
        }

        Ok(())
    }

    /// Get all non-zero elements in a row
    pub fn get_row_elements(&self, row: usize) -> TorshResult<Vec<(usize, f32)>> {
        if row >= self.shape.dims()[0] {
            return Err(TorshError::InvalidArgument(
                "Row index out of bounds".to_string(),
            ));
        }

        Ok(self.rows[row]
            .iter()
            .map(|(&col, &val)| (col, val))
            .collect())
    }

    /// Clear all elements in a row
    pub fn clear_row(&mut self, row: usize) -> TorshResult<()> {
        if row >= self.shape.dims()[0] {
            return Err(TorshError::InvalidArgument(
                "Row index out of bounds".to_string(),
            ));
        }

        let row_nnz = self.rows[row].len();
        self.rows[row].clear();
        self.cached_nnz -= row_nnz;

        Ok(())
    }

    /// Get number of non-zero elements in a specific row
    pub fn row_nnz(&self, row: usize) -> TorshResult<usize> {
        if row >= self.shape.dims()[0] {
            return Err(TorshError::InvalidArgument(
                "Row index out of bounds".to_string(),
            ));
        }

        Ok(self.rows[row].len())
    }

    /// Apply a function to modify all values in place
    pub fn apply_inplace<F>(&mut self, mut f: F) -> TorshResult<()>
    where
        F: FnMut(f32) -> f32,
    {
        let mut elements_to_remove = Vec::new();

        for (row_idx, row) in self.rows.iter_mut().enumerate() {
            let mut new_values = Vec::new();

            for (&col, &val) in row.iter() {
                let new_val = f(val);
                if new_val.abs() > 1e-12 {
                    new_values.push((col, new_val));
                } else {
                    elements_to_remove.push((row_idx, col));
                }
            }

            // Update values
            for (col, new_val) in new_values {
                row.insert(col, new_val);
            }
        }

        // Remove zero elements
        for (row_idx, col) in elements_to_remove {
            self.rows[row_idx].remove(&col);
            self.cached_nnz -= 1;
        }

        Ok(())
    }

    /// Transpose the DSR tensor (returns a new DSR tensor)
    pub fn transpose(&self) -> TorshResult<Self> {
        let transposed_shape = Shape::new(vec![self.shape.dims()[1], self.shape.dims()[0]]);
        let mut transposed = Self::new(transposed_shape, self.dtype)?;

        for (row_idx, row) in self.rows.iter().enumerate() {
            for (&col_idx, &value) in row.iter() {
                transposed.set(col_idx, row_idx, value)?;
            }
        }

        Ok(transposed)
    }

    /// Convert to triplets format (row, col, value)
    pub fn triplets(&self) -> Vec<(usize, usize, f32)> {
        let mut triplets = Vec::with_capacity(self.cached_nnz);

        for (row_idx, row) in self.rows.iter().enumerate() {
            for (&col_idx, &value) in row.iter() {
                triplets.push((row_idx, col_idx, value));
            }
        }

        triplets
    }

    /// Efficiently add another DSR tensor to this one
    pub fn add_dsr(&mut self, other: &DsrTensor) -> TorshResult<()> {
        if self.shape != *other.shape() {
            return Err(TorshError::InvalidArgument(
                "Shape mismatch for DSR addition".to_string(),
            ));
        }

        for (row_idx, other_row) in other.rows.iter().enumerate() {
            for (&col_idx, &other_value) in other_row.iter() {
                let current_value = self.rows[row_idx].get(&col_idx).copied().unwrap_or(0.0);
                let new_value = current_value + other_value;

                if new_value.abs() > 1e-12 {
                    let was_present = self.rows[row_idx].contains_key(&col_idx);
                    self.rows[row_idx].insert(col_idx, new_value);
                    if !was_present {
                        self.cached_nnz += 1;
                    }
                } else if self.rows[row_idx].remove(&col_idx).is_some() {
                    self.cached_nnz -= 1;
                }
            }
        }

        Ok(())
    }

    /// Scale all values by a scalar
    pub fn scale(&mut self, scalar: f32) -> TorshResult<()> {
        if scalar.abs() < 1e-12 {
            // Scaling by zero - clear all elements
            for row in &mut self.rows {
                row.clear();
            }
            self.cached_nnz = 0;
        } else {
            for row in &mut self.rows {
                for value in row.values_mut() {
                    *value *= scalar;
                }
            }
        }
        Ok(())
    }
}

impl SparseTensor for DsrTensor {
    fn format(&self) -> SparseFormat {
        SparseFormat::Csr // Using CSR as closest equivalent for now
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
        self.cached_nnz
    }

    fn to_dense(&self) -> TorshResult<Tensor> {
        use torsh_tensor::creation::zeros;

        let dense = zeros::<f32>(self.shape.dims())?;

        // Note: This is a simplified implementation
        // In practice, you'd want to efficiently set values in the dense tensor
        // For now, we return a zero tensor as placeholder

        Ok(dense)
    }

    fn to_coo(&self) -> TorshResult<CooTensor> {
        let triplets = self.triplets();
        let (rows, cols, vals): (Vec<_>, Vec<_>, Vec<_>) = triplets.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut rs, mut cs, mut vs), (r, c, v)| {
                rs.push(r);
                cs.push(c);
                vs.push(v);
                (rs, cs, vs)
            },
        );

        CooTensor::new(rows, cols, vals, self.shape.clone())
    }

    fn to_csr(&self) -> TorshResult<CsrTensor> {
        let coo = self.to_coo()?;
        CsrTensor::from_coo(&coo)
    }

    fn to_csc(&self) -> TorshResult<CscTensor> {
        let coo = self.to_coo()?;
        CscTensor::from_coo(&coo)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_dsr_creation() {
        let shape = Shape::new(vec![3, 4]);
        let dsr = DsrTensor::new(shape, DType::F32).unwrap();

        assert_eq!(dsr.nnz(), 0);
        assert_eq!(dsr.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_dsr_set_get() {
        let shape = Shape::new(vec![3, 4]);
        let mut dsr = DsrTensor::new(shape, DType::F32).unwrap();

        dsr.set(1, 2, 5.0).unwrap();
        assert_relative_eq!(dsr.get(1, 2).unwrap(), 5.0);
        assert_eq!(dsr.nnz(), 1);

        // Test overwrite
        dsr.set(1, 2, 10.0).unwrap();
        assert_relative_eq!(dsr.get(1, 2).unwrap(), 10.0);
        assert_eq!(dsr.nnz(), 1);

        // Test zero removal
        dsr.set(1, 2, 0.0).unwrap();
        assert_relative_eq!(dsr.get(1, 2).unwrap(), 0.0);
        assert_eq!(dsr.nnz(), 0);
    }

    #[test]
    fn test_dsr_dynamic_operations() {
        let shape = Shape::new(vec![3, 3]);
        let mut dsr = DsrTensor::new(shape, DType::F32).unwrap();

        // Insert elements
        dsr.set(0, 0, 1.0).unwrap();
        dsr.set(0, 2, 2.0).unwrap();
        dsr.set(1, 1, 3.0).unwrap();
        dsr.set(2, 0, 4.0).unwrap();

        assert_eq!(dsr.nnz(), 4);

        // Test row operations
        let row_0_elements = dsr.get_row_elements(0).unwrap();
        assert_eq!(row_0_elements, vec![(0, 1.0), (2, 2.0)]);

        assert_eq!(dsr.row_nnz(0).unwrap(), 2);
        assert_eq!(dsr.row_nnz(1).unwrap(), 1);

        // Clear a row
        dsr.clear_row(0).unwrap();
        assert_eq!(dsr.nnz(), 2);
        assert_eq!(dsr.row_nnz(0).unwrap(), 0);
    }

    #[test]
    fn test_dsr_conversions() {
        let shape = Shape::new(vec![3, 3]);
        let mut dsr = DsrTensor::new(shape, DType::F32).unwrap();

        dsr.set(0, 0, 1.0).unwrap();
        dsr.set(1, 1, 2.0).unwrap();
        dsr.set(2, 2, 3.0).unwrap();

        // Test conversion to COO
        let coo = dsr.to_coo().unwrap();
        assert_eq!(coo.nnz(), 3);

        // Test round-trip conversion
        let dsr2 = DsrTensor::from_coo(&coo).unwrap();
        assert_eq!(dsr2.nnz(), 3);
        assert_relative_eq!(dsr2.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(dsr2.get(1, 1).unwrap(), 2.0);
        assert_relative_eq!(dsr2.get(2, 2).unwrap(), 3.0);
    }

    #[test]
    fn test_dsr_transpose() {
        let shape = Shape::new(vec![2, 3]);
        let mut dsr = DsrTensor::new(shape, DType::F32).unwrap();

        dsr.set(0, 1, 5.0).unwrap();
        dsr.set(1, 2, 10.0).unwrap();

        let transposed = dsr.transpose().unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_relative_eq!(transposed.get(1, 0).unwrap(), 5.0);
        assert_relative_eq!(transposed.get(2, 1).unwrap(), 10.0);
    }

    #[test]
    fn test_dsr_addition() {
        let shape = Shape::new(vec![2, 2]);
        let mut dsr1 = DsrTensor::new(shape.clone(), DType::F32).unwrap();
        let mut dsr2 = DsrTensor::new(shape, DType::F32).unwrap();

        dsr1.set(0, 0, 1.0).unwrap();
        dsr1.set(1, 1, 2.0).unwrap();

        dsr2.set(0, 0, 3.0).unwrap();
        dsr2.set(0, 1, 4.0).unwrap();

        dsr1.add_dsr(&dsr2).unwrap();

        assert_relative_eq!(dsr1.get(0, 0).unwrap(), 4.0);
        assert_relative_eq!(dsr1.get(0, 1).unwrap(), 4.0);
        assert_relative_eq!(dsr1.get(1, 1).unwrap(), 2.0);
    }

    #[test]
    fn test_dsr_scaling() {
        let shape = Shape::new(vec![2, 2]);
        let mut dsr = DsrTensor::new(shape, DType::F32).unwrap();

        dsr.set(0, 0, 2.0).unwrap();
        dsr.set(1, 1, 4.0).unwrap();

        dsr.scale(0.5).unwrap();

        assert_relative_eq!(dsr.get(0, 0).unwrap(), 1.0);
        assert_relative_eq!(dsr.get(1, 1).unwrap(), 2.0);
        assert_eq!(dsr.nnz(), 2);

        // Scale by zero
        dsr.scale(0.0).unwrap();
        assert_eq!(dsr.nnz(), 0);
    }
}
