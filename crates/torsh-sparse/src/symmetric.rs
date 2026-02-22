//! Symmetric sparse tensor format
//!
//! This format efficiently stores symmetric sparse tensors by only storing
//! the upper or lower triangle, reducing memory usage by approximately half.

use crate::{CooTensor, CscTensor, CsrTensor, SparseFormat, SparseTensor, TorshResult};
use std::collections::HashMap;
use torsh_core::{device::DeviceType, DType, Shape, TorshError};
use torsh_tensor::Tensor;

/// Storage mode for symmetric matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetricMode {
    /// Store only the upper triangle (including diagonal)
    Upper,
    /// Store only the lower triangle (including diagonal)
    Lower,
}

/// Symmetric sparse tensor
///
/// This format stores only one triangle of a symmetric matrix, automatically
/// providing symmetric access to the data. This can reduce memory usage by
/// approximately 50% for symmetric matrices.
#[derive(Debug, Clone)]
pub struct SymmetricTensor {
    /// Underlying sparse tensor storing one triangle
    triangle: CsrTensor,
    /// Which triangle is stored
    mode: SymmetricMode,
    /// Shape of the full symmetric tensor
    shape: Shape,
    /// Data type
    dtype: DType,
    /// Device type
    device: DeviceType,
    /// Number of non-zero elements in the full symmetric matrix
    full_nnz: usize,
}

impl SymmetricTensor {
    /// Create a new symmetric tensor from a triangle
    pub fn new(
        triangle: CsrTensor,
        mode: SymmetricMode,
        shape: Shape,
        dtype: DType,
        device: DeviceType,
    ) -> TorshResult<Self> {
        // Validate that the shape is square
        if shape.dims().len() != 2 || shape.dims()[0] != shape.dims()[1] {
            return Err(TorshError::InvalidShape(
                "Symmetric tensors must be square matrices".to_string(),
            ));
        }

        // Calculate full nnz (triangle nnz + off-diagonal elements mirrored)
        let triangle_nnz = triangle.nnz();
        let diagonal_nnz = Self::count_diagonal_elements(&triangle);
        let full_nnz = triangle_nnz + (triangle_nnz - diagonal_nnz);

        Ok(Self {
            triangle,
            mode,
            shape,
            dtype,
            device,
            full_nnz,
        })
    }

    /// Create symmetric tensor from dense tensor
    pub fn from_dense(dense: &Tensor, mode: SymmetricMode, threshold: f32) -> TorshResult<Self> {
        let shape = dense.shape();
        let dtype = dense.dtype();
        let device = dense.device();

        if shape.dims().len() != 2 || shape.dims()[0] != shape.dims()[1] {
            return Err(TorshError::InvalidShape(
                "Symmetric tensors must be square matrices".to_string(),
            ));
        }

        let n = shape.dims()[0];
        let data = dense.to_vec()?;

        // Check if matrix is actually symmetric (within tolerance)
        if !Self::is_symmetric(&data, n, threshold * 0.1) {
            return Err(TorshError::InvalidShape(
                "Input matrix is not symmetric".to_string(),
            ));
        }

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();

        // Extract the specified triangle
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                let value = data[idx];

                if value.abs() > threshold {
                    let should_include = match mode {
                        SymmetricMode::Upper => j >= i,
                        SymmetricMode::Lower => j <= i,
                    };

                    if should_include {
                        rows.push(i);
                        cols.push(j);
                        values.push(value);
                    }
                }
            }
        }

        let triangle = CsrTensor::new(rows, cols, values, shape.clone())?;
        Self::new(triangle, mode, shape, dtype, device)
    }

    /// Create symmetric tensor from COO tensor
    pub fn from_coo(coo: &CooTensor, mode: SymmetricMode, threshold: f32) -> TorshResult<Self> {
        let shape = coo.shape().clone();
        let dtype = coo.dtype();
        let device = coo.device();

        if shape.dims().len() != 2 || shape.dims()[0] != shape.dims()[1] {
            return Err(TorshError::InvalidShape(
                "Symmetric tensors must be square matrices".to_string(),
            ));
        }

        let triplets = coo.triplets();

        // Check symmetry and extract triangle
        let mut triangle_rows = Vec::new();
        let mut triangle_cols = Vec::new();
        let mut triangle_values = Vec::new();

        // Create map for quick lookup of symmetric elements
        let mut element_map: HashMap<(usize, usize), f32> = HashMap::new();
        for (row, col, value) in triplets.iter() {
            element_map.insert((*row, *col), *value);
        }

        // Verify symmetry and extract triangle
        for (row, col, value) in triplets.iter() {
            if *row != *col {
                // Check if symmetric element exists and is approximately equal
                if let Some(&sym_value) = element_map.get(&(*col, *row)) {
                    if (value - sym_value).abs() > threshold * 0.1 {
                        return Err(TorshError::InvalidShape(
                            "Input matrix is not symmetric".to_string(),
                        ));
                    }
                } else {
                    return Err(TorshError::InvalidShape(
                        "Input matrix is not symmetric (missing element)".to_string(),
                    ));
                }
            }

            // Include element if it's in the desired triangle
            let should_include = match mode {
                SymmetricMode::Upper => *col >= *row,
                SymmetricMode::Lower => *col <= *row,
            };

            if should_include {
                triangle_rows.push(*row);
                triangle_cols.push(*col);
                triangle_values.push(*value);
            }
        }

        let triangle_coo =
            CooTensor::new(triangle_rows, triangle_cols, triangle_values, shape.clone())?;
        let triangle = CsrTensor::from_coo(&triangle_coo)?;
        Self::new(triangle, mode, shape, dtype, device)
    }

    /// Get the storage mode
    pub fn mode(&self) -> SymmetricMode {
        self.mode
    }

    /// Get the underlying triangle tensor
    pub fn triangle(&self) -> &CsrTensor {
        &self.triangle
    }

    /// Get value at specific position (handles symmetry automatically)
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        // First try direct lookup in stored triangle
        if let Some(value) = self.triangle.get(row, col) {
            return Some(value);
        }

        // If not found and not on diagonal, try symmetric position
        if row != col {
            if let Some(value) = self.triangle.get(col, row) {
                return Some(value);
            }
        }

        None
    }

    /// Check if a matrix is symmetric within tolerance
    fn is_symmetric(data: &[f32], n: usize, tolerance: f32) -> bool {
        for i in 0..n {
            for j in 0..n {
                let idx1 = i * n + j;
                let idx2 = j * n + i;
                if (data[idx1] - data[idx2]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Count diagonal elements in the triangle tensor
    fn count_diagonal_elements(triangle: &CsrTensor) -> usize {
        let triplets = triangle.triplets();
        triplets.iter().filter(|(r, c, _)| r == c).count()
    }

    /// Calculate memory savings compared to full storage
    pub fn memory_savings_ratio(&self) -> f32 {
        let full_storage = self.full_nnz;
        let triangle_storage = self.triangle.nnz();

        if full_storage == 0 {
            1.0
        } else {
            triangle_storage as f32 / full_storage as f32
        }
    }

    /// Create a full (non-symmetric) sparse tensor by expanding the triangle
    pub fn to_full_sparse(&self) -> TorshResult<CsrTensor> {
        let triplets = self.triangle.triplets();

        let mut full_rows = Vec::new();
        let mut full_cols = Vec::new();
        let mut full_values = Vec::new();

        // Add all triangle elements
        for (row, col, value) in triplets.iter() {
            full_rows.push(*row);
            full_cols.push(*col);
            full_values.push(*value);

            // Add symmetric element if not on diagonal
            if *row != *col {
                full_rows.push(*col);
                full_cols.push(*row);
                full_values.push(*value);
            }
        }

        // Use COO as intermediate format to properly construct CSR
        let coo = CooTensor::new(full_rows, full_cols, full_values, self.shape.clone())?;
        CsrTensor::from_coo(&coo)
    }
}

impl SparseTensor for SymmetricTensor {
    fn format(&self) -> SparseFormat {
        SparseFormat::Symmetric
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
        self.triangle.nnz()
    }

    fn to_dense(&self) -> TorshResult<Tensor> {
        let full_sparse = self.to_full_sparse()?;
        full_sparse.to_dense()
    }

    fn to_coo(&self) -> TorshResult<CooTensor> {
        let full_sparse = self.to_full_sparse()?;
        full_sparse.to_coo()
    }

    fn to_csr(&self) -> TorshResult<CsrTensor> {
        self.to_full_sparse()
    }

    fn to_csc(&self) -> TorshResult<CscTensor> {
        let full_sparse = self.to_full_sparse()?;
        full_sparse.to_csc()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_creation() {
        // Create a simple 2x2 symmetric matrix triangle
        // Matrix: [[1, 2], [2, 3]] upper triangle: [[1, 2], [_, 3]]
        let row_ptr = vec![0, 2, 3]; // Row 0 has 2 elements, row 1 has 1 element
        let cols = vec![0, 1, 1]; // Col indices: row 0 -> cols 0,1; row 1 -> col 1
        let values = vec![1.0, 2.0, 3.0];

        let shape = Shape::new(vec![2, 2]);
        let triangle = CsrTensor::new(row_ptr, cols, values, shape.clone()).unwrap();
        let sym = SymmetricTensor::new(
            triangle,
            SymmetricMode::Upper,
            shape,
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();

        assert_eq!(sym.nnz(), 3); // Only upper triangle: (0,0), (0,1), (1,1)
        assert_eq!(sym.mode(), SymmetricMode::Upper);
        assert_eq!(sym.get(0, 0), Some(1.0));
        assert_eq!(sym.get(0, 1), Some(2.0));
        assert_eq!(sym.get(1, 0), Some(2.0)); // Symmetric access
        assert_eq!(sym.get(1, 1), Some(3.0));
    }

    #[test]
    fn test_symmetric_memory_savings() {
        // Create a 3x3 upper triangular matrix
        // Matrix: [[1, 2, 0], [2, 3, 4], [0, 4, 5]]
        // Upper triangle: [[1, 2, 0], [_, 3, 4], [_, _, 5]]
        let row_ptr = vec![0, 2, 4, 5]; // Row 0: 2 elements, row 1: 2 elements, row 2: 1 element
        let cols = vec![0, 1, 1, 2, 2]; // Col indices
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let shape = Shape::new(vec![3, 3]);
        let triangle = CsrTensor::new(row_ptr, cols, values, shape.clone()).unwrap();
        let sym = SymmetricTensor::new(
            triangle,
            SymmetricMode::Upper,
            shape,
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();

        let savings = sym.memory_savings_ratio();
        assert!(savings < 1.0); // Should save memory
        assert!(savings > 0.5); // But not too much for this small example
    }

    #[test]
    fn test_symmetric_conversion() {
        // Create a 2x2 upper triangular matrix
        let row_ptr = vec![0, 2, 3]; // Row pointers
        let cols = vec![0, 1, 1];
        let values = vec![1.0, 2.0, 3.0];

        let shape = Shape::new(vec![2, 2]);
        let triangle = CsrTensor::new(row_ptr, cols, values, shape.clone()).unwrap();
        let sym = SymmetricTensor::new(
            triangle,
            SymmetricMode::Upper,
            shape,
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();

        let full_csr = sym.to_csr().unwrap();
        assert_eq!(full_csr.nnz(), 4); // Full matrix including symmetric elements

        let coo = sym.to_coo().unwrap();
        assert_eq!(coo.nnz(), 4);

        let dense = sym.to_dense().unwrap();
        assert_eq!(dense.shape(), sym.shape);
    }

    #[test]
    fn test_symmetric_mode_lower() {
        // Create a 2x2 lower triangular matrix
        // Matrix: [[1, 2], [2, 3]] lower triangle: [[1, _], [2, 3]]
        let row_ptr = vec![0, 1, 3]; // Row 0: 1 element, row 1: 2 elements
        let cols = vec![0, 0, 1]; // Col indices: row 0 -> col 0; row 1 -> cols 0,1
        let values = vec![1.0, 2.0, 3.0];

        let shape = Shape::new(vec![2, 2]);
        let triangle = CsrTensor::new(row_ptr, cols, values, shape.clone()).unwrap();
        let sym = SymmetricTensor::new(
            triangle,
            SymmetricMode::Lower,
            shape,
            DType::F32,
            DeviceType::Cpu,
        )
        .unwrap();

        assert_eq!(sym.mode(), SymmetricMode::Lower);
        assert_eq!(sym.get(0, 0), Some(1.0));
        assert_eq!(sym.get(1, 0), Some(2.0));
        assert_eq!(sym.get(0, 1), Some(2.0)); // Symmetric access
        assert_eq!(sym.get(1, 1), Some(3.0));
    }

    #[test]
    fn test_is_symmetric() {
        // Symmetric matrix: [[1, 2], [2, 3]]
        let symmetric_data = vec![1.0, 2.0, 2.0, 3.0];
        assert!(SymmetricTensor::is_symmetric(&symmetric_data, 2, 1e-6));

        // Non-symmetric matrix: [[1, 2], [3, 4]]
        let non_symmetric_data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(!SymmetricTensor::is_symmetric(&non_symmetric_data, 2, 1e-6));
    }
}
