//! Sparse tensor implementation with multiple storage formats
//!
//! This module provides efficient sparse tensor operations for tensors with many zero elements.
//! Supports three storage formats:
//! - COO (Coordinate): General purpose, easy to construct and modify
//! - CSR (Compressed Sparse Row): Efficient for row-wise operations and matrix-vector multiplication
//! - CSC (Compressed Sparse Column): Efficient for column-wise operations
//!
//! Each format has different performance characteristics:
//! - COO: Best for construction, random access, and format conversion
//! - CSR: Best for row slicing, matrix-vector multiplication (A*x)
//! - CSC: Best for column slicing, matrix-vector multiplication (A^T*x)

use crate::{core_ops::Tensor, TensorElement};
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use torsh_core::{
    device::DeviceType,
    error::{Result, TorshError},
}; // SciRS2 Parallel Operations for sparse tensor processing

/// Sparse tensor in COO (Coordinate) format
///
/// COO format stores sparse tensors efficiently by only keeping track of non-zero elements
/// and their coordinates. This is particularly useful for tensors with high sparsity ratios.
///
/// # Format
/// - `indices`: N x D matrix where N is the number of non-zero elements and D is the number of dimensions
/// - `values`: N-length vector containing the non-zero values
/// - `shape`: The shape of the full dense tensor
///
/// # Example
/// ```rust
/// use torsh_tensor::sparse::SparseTensor;
///
/// // Create a 3x3 sparse matrix with values at (0,0)=1.0, (1,2)=2.0, (2,1)=3.0
/// let indices = vec![vec![0, 0], vec![1, 2], vec![2, 1]];
/// let values = vec![1.0, 2.0, 3.0];
/// let shape = vec![3, 3];
///
/// let sparse = SparseTensor::from_coo(indices, values, shape).expect("sparse tensor creation should succeed");
/// ```
#[derive(Debug, Clone)]
pub struct SparseTensor<T: TensorElement> {
    /// Coordinates of non-zero elements (N x D matrix)
    indices: Vec<Vec<usize>>,
    /// Non-zero values (N elements)
    values: Vec<T>,
    /// Shape of the full dense tensor
    shape: Vec<usize>,
    /// Device where the tensor resides
    device: DeviceType,
    /// Number of non-zero elements
    nnz: usize,
}

impl<T: TensorElement> SparseTensor<T> {
    /// Create a new sparse tensor from COO format
    ///
    /// # Arguments
    /// * `indices` - Coordinates of non-zero elements (each inner vector is one coordinate)
    /// * `values` - Non-zero values corresponding to the indices
    /// * `shape` - Shape of the full dense tensor
    ///
    /// # Returns
    /// A new sparse tensor in COO format
    ///
    /// # Errors
    /// Returns error if indices and values have mismatched lengths or if coordinates are out of bounds
    pub fn from_coo(indices: Vec<Vec<usize>>, values: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Indices length ({}) must match values length ({})",
                indices.len(),
                values.len()
            )));
        }

        let ndim = shape.len();
        for (i, coord) in indices.iter().enumerate() {
            if coord.len() != ndim {
                return Err(TorshError::InvalidArgument(format!(
                    "Index {} has {} dimensions, expected {}",
                    i,
                    coord.len(),
                    ndim
                )));
            }

            for (dim, &idx) in coord.iter().enumerate() {
                if idx >= shape[dim] {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {} at dimension {} is out of bounds ({})",
                        idx, dim, shape[dim]
                    )));
                }
            }
        }

        Ok(Self {
            nnz: indices.len(),
            indices,
            values,
            shape,
            device: DeviceType::Cpu,
        })
    }

    /// Create a sparse tensor from a dense tensor by extracting non-zero elements
    ///
    /// # Arguments
    /// * `dense` - The dense tensor to convert
    /// * `tolerance` - Values with absolute value below this threshold are considered zero
    ///
    /// # Returns
    /// A new sparse tensor containing only the non-zero elements
    pub fn from_dense(dense: &Tensor<T>, tolerance: T) -> Result<Self>
    where
        T: Copy + PartialOrd + num_traits::Zero + num_traits::Signed,
    {
        let data = dense.data()?;
        let shape = dense.shape().dims().to_vec();

        let mut indices = Vec::new();
        let mut values = Vec::new();

        // Iterate through all elements and collect values above tolerance
        for flat_idx in 0..data.len() {
            let value = data[flat_idx];

            // Check if absolute value exceeds tolerance threshold
            let abs_value = value.abs();

            // Only include values that exceed the tolerance threshold
            if abs_value > tolerance {
                // Convert flat index to multi-dimensional coordinates
                let coords = Self::flat_to_coords(flat_idx, &shape);
                indices.push(coords);
                values.push(value);
            }
        }

        Self::from_coo(indices, values, shape)
    }

    /// Convert this sparse tensor to a dense tensor
    ///
    /// # Returns
    /// A dense tensor with zeros filled in for missing elements
    pub fn to_dense(&self) -> Result<Tensor<T>>
    where
        T: Copy + num_traits::Zero,
    {
        let total_elements: usize = self.shape.iter().product();
        let mut data = vec![<T as num_traits::Zero>::zero(); total_elements];

        // Fill in the non-zero values
        for (coords, &value) in self.indices.iter().zip(self.values.iter()) {
            let flat_idx = Self::coords_to_flat(coords, &self.shape);
            data[flat_idx] = value;
        }

        Tensor::from_data(data, self.shape.clone(), self.device)
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Get the shape of the sparse tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the device where the tensor resides
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Get the indices (coordinates) of non-zero elements
    pub fn indices(&self) -> &[Vec<usize>] {
        &self.indices
    }

    /// Get the non-zero values
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Calculate the sparsity ratio (fraction of zero elements)
    pub fn sparsity(&self) -> f64 {
        let total_elements: usize = self.shape.iter().product();
        1.0 - (self.nnz as f64 / total_elements as f64)
    }

    /// Get the memory footprint of the sparse representation in bytes
    pub fn memory_usage(&self) -> usize {
        let indices_size = self.indices.len() * self.shape.len() * std::mem::size_of::<usize>();
        let values_size = self.values.len() * std::mem::size_of::<T>();
        let shape_size = self.shape.len() * std::mem::size_of::<usize>();

        indices_size + values_size + shape_size + std::mem::size_of::<Self>()
    }

    /// Compare memory usage with equivalent dense representation
    pub fn memory_efficiency(&self) -> f64 {
        let total_elements: usize = self.shape.iter().product();
        let dense_size = total_elements * std::mem::size_of::<T>();
        let sparse_size = self.memory_usage();

        1.0 - (sparse_size as f64 / dense_size as f64)
    }

    /// Element-wise addition with another sparse tensor
    ///
    /// # Arguments
    /// * `other` - The other sparse tensor to add
    ///
    /// # Returns
    /// A new sparse tensor containing the sum
    pub fn add(&self, other: &Self) -> Result<Self>
    where
        T: Copy + std::ops::Add<Output = T> + num_traits::Zero + PartialEq,
    {
        if self.shape != other.shape {
            return Err(TorshError::InvalidArgument(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        // Use HashMap to efficiently merge non-zero elements
        let mut result_map: HashMap<Vec<usize>, T> = HashMap::new();

        // Add elements from first tensor
        for (coords, &value) in self.indices.iter().zip(self.values.iter()) {
            result_map.insert(coords.clone(), value);
        }

        // Add elements from second tensor
        for (coords, &value) in other.indices.iter().zip(other.values.iter()) {
            match result_map.get_mut(coords) {
                Some(existing_value) => {
                    *existing_value = *existing_value + value;
                }
                None => {
                    result_map.insert(coords.clone(), value);
                }
            }
        }

        // Filter out zeros and collect results
        let zero = <T as num_traits::Zero>::zero();
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (coords, value) in result_map {
            if value != zero {
                indices.push(coords);
                values.push(value);
            }
        }

        Self::from_coo(indices, values, self.shape.clone())
    }

    /// Element-wise multiplication with another sparse tensor
    ///
    /// For sparse tensors, multiplication only produces non-zero results where both tensors
    /// have non-zero elements at the same location.
    pub fn mul(&self, other: &Self) -> Result<Self>
    where
        T: Copy + std::ops::Mul<Output = T> + num_traits::Zero + PartialEq,
    {
        if self.shape != other.shape {
            return Err(TorshError::InvalidArgument(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        // Create HashMap for efficient lookup
        let other_map: HashMap<Vec<usize>, T> = other
            .indices
            .iter()
            .zip(other.values.iter())
            .map(|(coords, &value)| (coords.clone(), value))
            .collect();

        let mut indices = Vec::new();
        let mut values = Vec::new();
        let zero = <T as num_traits::Zero>::zero();

        // Only multiply where both tensors have non-zero elements
        for (coords, &value) in self.indices.iter().zip(self.values.iter()) {
            if let Some(&other_value) = other_map.get(coords) {
                let result = value * other_value;
                if result != zero {
                    indices.push(coords.clone());
                    values.push(result);
                }
            }
        }

        Self::from_coo(indices, values, self.shape.clone())
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: T) -> Result<Self>
    where
        T: Copy + std::ops::Mul<Output = T> + num_traits::Zero + PartialEq,
    {
        let zero = <T as num_traits::Zero>::zero();
        if scalar == zero {
            // Result is all zeros - return empty sparse tensor
            return Self::from_coo(Vec::new(), Vec::new(), self.shape.clone());
        }

        let new_values: Vec<T> = self.values.iter().map(|&v| v * scalar).collect();

        Self::from_coo(self.indices.clone(), new_values, self.shape.clone())
    }

    /// Matrix multiplication for 2D sparse tensors
    ///
    /// # Arguments
    /// * `other` - The other 2D sparse tensor to multiply with
    ///
    /// # Returns
    /// A new sparse tensor containing the matrix product
    pub fn matmul(&self, other: &Self) -> Result<Self>
    where
        T: Copy
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + num_traits::Zero
            + PartialEq,
    {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        if self.shape[1] != other.shape[0] {
            return Err(TorshError::InvalidArgument(format!(
                "Incompatible shapes for matmul: {:?} x {:?}",
                self.shape, other.shape
            )));
        }

        let m = self.shape[0];
        let n = other.shape[1];
        let k = self.shape[1];

        // Sparse matrix multiplication uses only non-zero entries
        // k (inner dimension) is validated for compatibility but not directly iterated
        let _ = (m, k, n); // Use dimensions for validation

        // Create efficient lookup structures
        let mut left_rows: HashMap<usize, Vec<(usize, T)>> = HashMap::new();
        let mut right_cols: HashMap<usize, Vec<(usize, T)>> = HashMap::new();

        // Organize left matrix by rows
        for (coords, &value) in self.indices.iter().zip(self.values.iter()) {
            let row = coords[0];
            let col = coords[1];
            left_rows
                .entry(row)
                .or_insert_with(Vec::new)
                .push((col, value));
        }

        // Organize right matrix by columns
        for (coords, &value) in other.indices.iter().zip(other.values.iter()) {
            let row = coords[0];
            let col = coords[1];
            right_cols
                .entry(col)
                .or_insert_with(Vec::new)
                .push((row, value));
        }

        let mut result_map: HashMap<Vec<usize>, T> = HashMap::new();
        let zero = <T as num_traits::Zero>::zero();

        // Compute matrix multiplication
        for (&row, left_row_data) in left_rows.iter() {
            for (&col, right_col_data) in right_cols.iter() {
                let mut sum = zero;

                // Compute dot product of row and column
                let mut left_iter = left_row_data.iter().peekable();
                let mut right_iter = right_col_data.iter().peekable();

                while let (Some(&(left_col, left_val)), Some(&(right_row, right_val))) =
                    (left_iter.peek(), right_iter.peek())
                {
                    match left_col.cmp(&right_row) {
                        std::cmp::Ordering::Equal => {
                            sum = sum + (*left_val) * (*right_val);
                            left_iter.next();
                            right_iter.next();
                        }
                        std::cmp::Ordering::Less => {
                            left_iter.next();
                        }
                        std::cmp::Ordering::Greater => {
                            right_iter.next();
                        }
                    }
                }

                if sum != zero {
                    result_map.insert(vec![row, col], sum);
                }
            }
        }

        // Convert result map to COO format
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (coords, value) in result_map {
            indices.push(coords);
            values.push(value);
        }

        Self::from_coo(indices, values, vec![m, n])
    }

    /// Convert flat index to multi-dimensional coordinates
    fn flat_to_coords(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; shape.len()];
        let mut remaining = flat_idx;

        for i in 0..shape.len() {
            let stride: usize = shape[i + 1..].iter().product();
            coords[i] = remaining / stride;
            remaining %= stride;
        }

        coords
    }

    /// Convert multi-dimensional coordinates to flat index
    fn coords_to_flat(coords: &[usize], shape: &[usize]) -> usize {
        let mut flat_idx = 0;
        let mut stride = 1;

        for i in (0..coords.len()).rev() {
            flat_idx += coords[i] * stride;
            stride *= shape[i];
        }

        flat_idx
    }

    /// Transpose a 2D sparse tensor
    pub fn transpose(&self) -> Result<Self>
    where
        T: Copy,
    {
        if self.shape.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Transpose is only supported for 2D tensors".to_string(),
            ));
        }

        let new_shape = vec![self.shape[1], self.shape[0]];
        let new_indices: Vec<Vec<usize>> = self
            .indices
            .iter()
            .map(|coords| vec![coords[1], coords[0]])
            .collect();

        Self::from_coo(new_indices, self.values.clone(), new_shape)
    }

    /// Apply a function to all non-zero values
    pub fn map<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(T) -> T,
        T: Copy + num_traits::Zero + PartialEq,
    {
        let new_values: Vec<T> = self.values.iter().map(|&v| f(v)).collect();

        // Filter out any values that became zero
        let zero = <T as num_traits::Zero>::zero();
        let mut filtered_indices = Vec::new();
        let mut filtered_values = Vec::new();

        for (coords, &value) in self.indices.iter().zip(new_values.iter()) {
            if value != zero {
                filtered_indices.push(coords.clone());
                filtered_values.push(value);
            }
        }

        Self::from_coo(filtered_indices, filtered_values, self.shape.clone())
    }

    /// Check if the sparse tensor is structurally valid
    pub fn is_valid(&self) -> bool {
        // Check that indices and values have same length
        if self.indices.len() != self.values.len() {
            return false;
        }

        // Check that nnz matches actual length
        if self.nnz != self.indices.len() {
            return false;
        }

        // Check that all indices are within bounds
        let ndim = self.shape.len();
        for coords in &self.indices {
            if coords.len() != ndim {
                return false;
            }

            for (dim, &idx) in coords.iter().enumerate() {
                if idx >= self.shape[dim] {
                    return false;
                }
            }
        }

        true
    }

    /// Remove duplicate indices by summing their values
    pub fn coalesce(&mut self) -> Result<()>
    where
        T: Copy + std::ops::AddAssign + num_traits::Zero + PartialEq,
    {
        if self.indices.is_empty() {
            return Ok(());
        }

        let mut coord_map: HashMap<Vec<usize>, T> = HashMap::new();

        // Sum values for duplicate coordinates
        for (coords, &value) in self.indices.iter().zip(self.values.iter()) {
            match coord_map.get_mut(coords) {
                Some(existing_value) => {
                    *existing_value += value;
                }
                None => {
                    coord_map.insert(coords.clone(), value);
                }
            }
        }

        // Filter out zeros and rebuild indices/values
        let zero = <T as num_traits::Zero>::zero();
        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        for (coords, value) in coord_map {
            if value != zero {
                new_indices.push(coords);
                new_values.push(value);
            }
        }

        self.indices = new_indices;
        self.values = new_values;
        self.nnz = self.indices.len();

        Ok(())
    }
}

/// Conversion utilities for sparse tensors
impl<T: TensorElement> SparseTensor<T> {
    /// Create a sparse identity matrix
    pub fn eye(size: usize) -> Result<Self>
    where
        T: Copy + num_traits::One,
    {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let one = <T as num_traits::One>::one();

        for i in 0..size {
            indices.push(vec![i, i]);
            values.push(one);
        }

        Self::from_coo(indices, values, vec![size, size])
    }

    /// Create a sparse tensor from triplets (row, col, value) for 2D case
    pub fn from_triplets(
        rows: Vec<usize>,
        cols: Vec<usize>,
        vals: Vec<T>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        if rows.len() != cols.len() || cols.len() != vals.len() {
            return Err(TorshError::InvalidArgument(
                "Rows, cols, and values must have the same length".to_string(),
            ));
        }

        let indices: Vec<Vec<usize>> = rows
            .into_iter()
            .zip(cols.into_iter())
            .map(|(r, c)| vec![r, c])
            .collect();

        Self::from_coo(indices, vals, shape)
    }
}

/// Sparse tensor in CSR (Compressed Sparse Row) format
///
/// CSR format is optimized for row-wise operations and matrix-vector multiplication.
/// It stores:
/// - `row_ptr`: Array of size (num_rows + 1) indicating where each row starts
/// - `col_indices`: Column indices for each non-zero value
/// - `values`: Non-zero values in row-major order
///
/// # Example
/// ```rust
/// use torsh_tensor::sparse::SparseCSR;
///
/// // Create a 3x3 sparse matrix in CSR format
/// // [[1.0, 0.0, 2.0],
/// //  [0.0, 3.0, 0.0],
/// //  [4.0, 0.0, 5.0]]
/// let row_ptr = vec![0, 2, 3, 5];  // Row pointers
/// let col_indices = vec![0, 2, 1, 0, 2];  // Column indices
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];  // Values
/// let shape = vec![3, 3];
///
/// let sparse = SparseCSR::new(row_ptr, col_indices, values, shape).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparseCSR<T: TensorElement> {
    /// Row pointers (size: num_rows + 1)
    row_ptr: Vec<usize>,
    /// Column indices for each non-zero value
    col_indices: Vec<usize>,
    /// Non-zero values in row-major order
    values: Vec<T>,
    /// Shape of the full dense tensor [num_rows, num_cols]
    shape: Vec<usize>,
    /// Device where the tensor resides
    device: DeviceType,
    /// Number of non-zero elements
    nnz: usize,
}

impl<T: TensorElement> SparseCSR<T> {
    /// Create a new CSR sparse tensor
    ///
    /// # Arguments
    /// * `row_ptr` - Row pointers (length: num_rows + 1)
    /// * `col_indices` - Column indices for non-zero values
    /// * `values` - Non-zero values
    /// * `shape` - Shape [num_rows, num_cols]
    pub fn new(
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        if shape.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CSR format only supports 2D tensors".to_string(),
            ));
        }

        if col_indices.len() != values.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Column indices length ({}) must match values length ({})",
                col_indices.len(),
                values.len()
            )));
        }

        if row_ptr.len() != shape[0] + 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Row pointer length ({}) must be num_rows + 1 ({})",
                row_ptr.len(),
                shape[0] + 1
            )));
        }

        // Validate row pointers are monotonically increasing
        for i in 1..row_ptr.len() {
            if row_ptr[i] < row_ptr[i - 1] {
                return Err(TorshError::InvalidArgument(
                    "Row pointers must be monotonically increasing".to_string(),
                ));
            }
        }

        // Validate column indices are within bounds
        for &col_idx in &col_indices {
            if col_idx >= shape[1] {
                return Err(TorshError::InvalidArgument(format!(
                    "Column index {} out of bounds for shape {:?}",
                    col_idx, shape
                )));
            }
        }

        let nnz = values.len();
        if row_ptr.last().copied().unwrap_or(0) != nnz {
            return Err(TorshError::InvalidArgument(
                "Last row pointer must equal number of non-zero values".to_string(),
            ));
        }

        Ok(Self {
            row_ptr,
            col_indices,
            values,
            shape,
            device: DeviceType::Cpu,
            nnz,
        })
    }

    /// Convert from COO format to CSR format
    pub fn from_coo(coo: &SparseTensor<T>) -> Result<Self>
    where
        T: Copy,
    {
        if coo.shape().len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CSR format only supports 2D tensors".to_string(),
            ));
        }

        let num_rows = coo.shape()[0];
        let num_cols = coo.shape()[1];

        // Sort COO entries by row, then column
        let mut entries: Vec<(usize, usize, T)> = coo
            .indices()
            .iter()
            .zip(coo.values())
            .map(|(coords, &val)| (coords[0], coords[1], val))
            .collect();

        entries.sort_by(|a, b| {
            if a.0 == b.0 {
                a.1.cmp(&b.1)
            } else {
                a.0.cmp(&b.0)
            }
        });

        // Build CSR structure
        let mut row_ptr = vec![0; num_rows + 1];
        let mut col_indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());

        for (row, col, val) in entries {
            col_indices.push(col);
            values.push(val);
            row_ptr[row + 1] += 1;
        }

        // Convert counts to cumulative sum
        for i in 1..=num_rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        Self::new(row_ptr, col_indices, values, vec![num_rows, num_cols])
    }

    /// Convert to COO format
    pub fn to_coo(&self) -> Result<SparseTensor<T>>
    where
        T: Copy,
    {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..self.shape[0] {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for idx in start..end {
                indices.push(vec![row, self.col_indices[idx]]);
                values.push(self.values[idx]);
            }
        }

        SparseTensor::from_coo(indices, values, self.shape.clone())
    }

    /// Convert to dense tensor
    pub fn to_dense(&self) -> Result<Tensor<T>>
    where
        T: Copy + num_traits::Zero,
    {
        let total_elements = self.shape[0] * self.shape[1];
        let mut data = vec![<T as num_traits::Zero>::zero(); total_elements];

        for row in 0..self.shape[0] {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for idx in start..end {
                let col = self.col_indices[idx];
                let flat_idx = row * self.shape[1] + col;
                data[flat_idx] = self.values[idx];
            }
        }

        Tensor::from_data(data, self.shape.clone(), self.device)
    }

    /// Matrix-vector multiplication (optimized for CSR)
    pub fn matvec(&self, vec: &[T]) -> Result<Vec<T>>
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::Zero,
    {
        if vec.len() != self.shape[1] {
            return Err(TorshError::InvalidArgument(format!(
                "Vector length ({}) must match number of columns ({})",
                vec.len(),
                self.shape[1]
            )));
        }

        let mut result = vec![<T as num_traits::Zero>::zero(); self.shape[0]];

        // Parallel row-wise computation using SciRS2
        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, result_val)| {
                let start = self.row_ptr[row];
                let end = self.row_ptr[row + 1];
                let mut sum = <T as num_traits::Zero>::zero();

                for idx in start..end {
                    let col = self.col_indices[idx];
                    sum = sum + self.values[idx] * vec[col];
                }

                *result_val = sum;
            });

        Ok(result)
    }

    /// Get a specific row as a sparse vector
    pub fn get_row(&self, row: usize) -> Result<(Vec<usize>, Vec<T>)>
    where
        T: Copy,
    {
        if row >= self.shape[0] {
            return Err(TorshError::InvalidArgument(format!(
                "Row {} out of bounds for shape {:?}",
                row, self.shape
            )));
        }

        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        let col_indices = self.col_indices[start..end].to_vec();
        let values = self.values[start..end].to_vec();

        Ok((col_indices, values))
    }

    /// Getters
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn device(&self) -> DeviceType {
        self.device
    }

    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }
}

/// Sparse tensor in CSC (Compressed Sparse Column) format
///
/// CSC format is optimized for column-wise operations.
/// It stores:
/// - `col_ptr`: Array of size (num_cols + 1) indicating where each column starts
/// - `row_indices`: Row indices for each non-zero value
/// - `values`: Non-zero values in column-major order
///
/// # Example
/// ```rust
/// use torsh_tensor::sparse::SparseCSC;
///
/// // Create a 3x3 sparse matrix in CSC format
/// // [[1.0, 0.0, 2.0],
/// //  [0.0, 3.0, 0.0],
/// //  [4.0, 0.0, 5.0]]
/// let col_ptr = vec![0, 2, 3, 5];  // Column pointers
/// let row_indices = vec![0, 2, 1, 0, 2];  // Row indices
/// let values = vec![1.0, 4.0, 3.0, 2.0, 5.0];  // Values
/// let shape = vec![3, 3];
///
/// let sparse = SparseCSC::new(col_ptr, row_indices, values, shape).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparseCSC<T: TensorElement> {
    /// Column pointers (size: num_cols + 1)
    col_ptr: Vec<usize>,
    /// Row indices for each non-zero value
    row_indices: Vec<usize>,
    /// Non-zero values in column-major order
    values: Vec<T>,
    /// Shape of the full dense tensor [num_rows, num_cols]
    shape: Vec<usize>,
    /// Device where the tensor resides
    device: DeviceType,
    /// Number of non-zero elements
    nnz: usize,
}

impl<T: TensorElement> SparseCSC<T> {
    /// Create a new CSC sparse tensor
    ///
    /// # Arguments
    /// * `col_ptr` - Column pointers (length: num_cols + 1)
    /// * `row_indices` - Row indices for non-zero values
    /// * `values` - Non-zero values
    /// * `shape` - Shape [num_rows, num_cols]
    pub fn new(
        col_ptr: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        if shape.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CSC format only supports 2D tensors".to_string(),
            ));
        }

        if row_indices.len() != values.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Row indices length ({}) must match values length ({})",
                row_indices.len(),
                values.len()
            )));
        }

        if col_ptr.len() != shape[1] + 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Column pointer length ({}) must be num_cols + 1 ({})",
                col_ptr.len(),
                shape[1] + 1
            )));
        }

        // Validate column pointers are monotonically increasing
        for i in 1..col_ptr.len() {
            if col_ptr[i] < col_ptr[i - 1] {
                return Err(TorshError::InvalidArgument(
                    "Column pointers must be monotonically increasing".to_string(),
                ));
            }
        }

        // Validate row indices are within bounds
        for &row_idx in &row_indices {
            if row_idx >= shape[0] {
                return Err(TorshError::InvalidArgument(format!(
                    "Row index {} out of bounds for shape {:?}",
                    row_idx, shape
                )));
            }
        }

        let nnz = values.len();
        if col_ptr.last().copied().unwrap_or(0) != nnz {
            return Err(TorshError::InvalidArgument(
                "Last column pointer must equal number of non-zero values".to_string(),
            ));
        }

        Ok(Self {
            col_ptr,
            row_indices,
            values,
            shape,
            device: DeviceType::Cpu,
            nnz,
        })
    }

    /// Convert from COO format to CSC format
    pub fn from_coo(coo: &SparseTensor<T>) -> Result<Self>
    where
        T: Copy,
    {
        if coo.shape().len() != 2 {
            return Err(TorshError::InvalidArgument(
                "CSC format only supports 2D tensors".to_string(),
            ));
        }

        let num_rows = coo.shape()[0];
        let num_cols = coo.shape()[1];

        // Sort COO entries by column, then row
        let mut entries: Vec<(usize, usize, T)> = coo
            .indices()
            .iter()
            .zip(coo.values())
            .map(|(coords, &val)| (coords[0], coords[1], val))
            .collect();

        entries.sort_by(|a, b| {
            if a.1 == b.1 {
                a.0.cmp(&b.0)
            } else {
                a.1.cmp(&b.1)
            }
        });

        // Build CSC structure
        let mut col_ptr = vec![0; num_cols + 1];
        let mut row_indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());

        for (row, col, val) in entries {
            row_indices.push(row);
            values.push(val);
            col_ptr[col + 1] += 1;
        }

        // Convert counts to cumulative sum
        for i in 1..=num_cols {
            col_ptr[i] += col_ptr[i - 1];
        }

        Self::new(col_ptr, row_indices, values, vec![num_rows, num_cols])
    }

    /// Convert to COO format
    pub fn to_coo(&self) -> Result<SparseTensor<T>>
    where
        T: Copy,
    {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for col in 0..self.shape[1] {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];

            for idx in start..end {
                indices.push(vec![self.row_indices[idx], col]);
                values.push(self.values[idx]);
            }
        }

        SparseTensor::from_coo(indices, values, self.shape.clone())
    }

    /// Convert to dense tensor
    pub fn to_dense(&self) -> Result<Tensor<T>>
    where
        T: Copy + num_traits::Zero,
    {
        let total_elements = self.shape[0] * self.shape[1];
        let mut data = vec![<T as num_traits::Zero>::zero(); total_elements];

        for col in 0..self.shape[1] {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];

            for idx in start..end {
                let row = self.row_indices[idx];
                let flat_idx = row * self.shape[1] + col;
                data[flat_idx] = self.values[idx];
            }
        }

        Tensor::from_data(data, self.shape.clone(), self.device)
    }

    /// Matrix-vector multiplication with transposed matrix (A^T * v) - optimized for CSC
    pub fn transpose_matvec(&self, vec: &[T]) -> Result<Vec<T>>
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::Zero,
    {
        if vec.len() != self.shape[0] {
            return Err(TorshError::InvalidArgument(format!(
                "Vector length ({}) must match number of rows ({})",
                vec.len(),
                self.shape[0]
            )));
        }

        let mut result = vec![<T as num_traits::Zero>::zero(); self.shape[1]];

        // Parallel column-wise computation using SciRS2
        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(col, result_val)| {
                let start = self.col_ptr[col];
                let end = self.col_ptr[col + 1];
                let mut sum = <T as num_traits::Zero>::zero();

                for idx in start..end {
                    let row = self.row_indices[idx];
                    sum = sum + self.values[idx] * vec[row];
                }

                *result_val = sum;
            });

        Ok(result)
    }

    /// Get a specific column as a sparse vector
    pub fn get_col(&self, col: usize) -> Result<(Vec<usize>, Vec<T>)>
    where
        T: Copy,
    {
        if col >= self.shape[1] {
            return Err(TorshError::InvalidArgument(format!(
                "Column {} out of bounds for shape {:?}",
                col, self.shape
            )));
        }

        let start = self.col_ptr[col];
        let end = self.col_ptr[col + 1];

        let row_indices = self.row_indices[start..end].to_vec();
        let values = self.values[start..end].to_vec();

        Ok((row_indices, values))
    }

    /// Getters
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn device(&self) -> DeviceType {
        self.device
    }

    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_sparse_tensor_creation() {
        let indices = vec![vec![0, 0], vec![1, 2], vec![2, 1]];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 3];

        let sparse = SparseTensor::from_coo(indices, values, shape).unwrap();
        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.shape(), &[3, 3]);
        assert!(sparse.sparsity() > 0.6); // 6 out of 9 elements are zero
    }

    #[test]
    fn test_sparse_to_dense_conversion() {
        let indices = vec![vec![0, 0], vec![1, 1], vec![2, 2]];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 3];

        let sparse = SparseTensor::from_coo(indices, values, shape).unwrap();
        let dense = sparse.to_dense().unwrap();

        let expected_data = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];

        assert_eq!(dense.data().unwrap(), expected_data);
    }

    #[test]
    fn test_sparse_addition() {
        let indices1 = vec![vec![0, 0], vec![1, 1]];
        let values1 = vec![1.0, 2.0];
        let shape = vec![3, 3];
        let sparse1 = SparseTensor::from_coo(indices1, values1, shape.clone()).unwrap();

        let indices2 = vec![vec![0, 0], vec![2, 2]];
        let values2 = vec![3.0, 4.0];
        let sparse2 = SparseTensor::from_coo(indices2, values2, shape).unwrap();

        let result = sparse1.add(&sparse2).unwrap();

        // Should have (0,0)=4.0, (1,1)=2.0, (2,2)=4.0
        assert_eq!(result.nnz(), 3);

        let dense_result = result.to_dense().unwrap();
        let expected = vec![4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0];
        assert_eq!(dense_result.data().unwrap(), expected);
    }

    #[test]
    fn test_sparse_multiplication() {
        let indices1 = vec![vec![0, 0], vec![1, 1], vec![2, 2]];
        let values1 = vec![2.0, 3.0, 4.0];
        let shape = vec![3, 3];
        let sparse1 = SparseTensor::from_coo(indices1, values1, shape.clone()).unwrap();

        let indices2 = vec![vec![0, 0], vec![1, 1]];
        let values2 = vec![5.0, 6.0];
        let sparse2 = SparseTensor::from_coo(indices2, values2, shape).unwrap();

        let result = sparse1.mul(&sparse2).unwrap();

        // Should have (0,0)=10.0, (1,1)=18.0
        assert_eq!(result.nnz(), 2);

        let dense_result = result.to_dense().unwrap();
        let expected = vec![10.0, 0.0, 0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(dense_result.data().unwrap(), expected);
    }

    #[test]
    fn test_sparse_matmul() {
        // Create a 2x2 sparse matrix [[1, 0], [0, 2]]
        let indices1 = vec![vec![0, 0], vec![1, 1]];
        let values1 = vec![1.0, 2.0];
        let shape1 = vec![2, 2];
        let sparse1 = SparseTensor::from_coo(indices1, values1, shape1).unwrap();

        // Create a 2x2 sparse matrix [[3, 0], [0, 4]]
        let indices2 = vec![vec![0, 0], vec![1, 1]];
        let values2 = vec![3.0, 4.0];
        let shape2 = vec![2, 2];
        let sparse2 = SparseTensor::from_coo(indices2, values2, shape2).unwrap();

        let result = sparse1.matmul(&sparse2).unwrap();

        // Result should be [[3, 0], [0, 8]]
        assert_eq!(result.nnz(), 2);

        let dense_result = result.to_dense().unwrap();
        let expected = vec![3.0, 0.0, 0.0, 8.0];
        assert_eq!(dense_result.data().unwrap(), expected);
    }

    #[test]
    fn test_sparse_transpose() {
        let indices = vec![vec![0, 1], vec![1, 0], vec![2, 1]];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 2];
        let sparse = SparseTensor::from_coo(indices, values, shape).unwrap();

        let transposed = sparse.transpose().unwrap();
        assert_eq!(transposed.shape(), &[2, 3]);

        let dense_transposed = transposed.to_dense().unwrap();
        let expected = vec![0.0, 2.0, 0.0, 1.0, 0.0, 3.0];
        assert_eq!(dense_transposed.data().unwrap(), expected);
    }

    #[test]
    fn test_sparse_identity() {
        let eye = SparseTensor::<f32>::eye(3).unwrap();
        assert_eq!(eye.nnz(), 3);
        assert_eq!(eye.shape(), &[3, 3]);

        let dense_eye = eye.to_dense().unwrap();
        let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(dense_eye.data().unwrap(), expected);
    }

    #[test]
    fn test_memory_efficiency() {
        let indices = vec![vec![0, 0]]; // Only one non-zero element
        let values = vec![1.0];
        let shape = vec![1000, 1000]; // Large tensor
        let sparse = SparseTensor::from_coo(indices, values, shape).unwrap();

        assert!(sparse.sparsity() > 0.999); // Very sparse
        assert!(sparse.memory_efficiency() > 0.9); // Much more memory efficient
    }

    #[test]
    fn test_from_dense() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let dense = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu).unwrap();

        let sparse = SparseTensor::from_dense(&dense, 1e-6).unwrap();
        assert_eq!(sparse.nnz(), 2);

        let back_to_dense = sparse.to_dense().unwrap();
        assert_eq!(dense.data().unwrap(), back_to_dense.data().unwrap());
    }

    #[test]
    fn test_coalesce() {
        // Create sparse tensor with duplicate indices
        let indices = vec![vec![0, 0], vec![1, 1], vec![0, 0]]; // (0,0) appears twice
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 2];

        let mut sparse = SparseTensor::from_coo(indices, values, shape).unwrap();
        assert_eq!(sparse.nnz(), 3);

        sparse.coalesce().unwrap();
        assert_eq!(sparse.nnz(), 2); // Should have combined duplicates

        let dense = sparse.to_dense().unwrap();
        let expected = vec![4.0, 0.0, 0.0, 2.0]; // (0,0) = 1+3 = 4
        assert_eq!(dense.data().unwrap(), expected);
    }

    #[test]
    fn test_scalar_multiplication() {
        let indices = vec![vec![0, 0], vec![1, 1]];
        let values = vec![2.0, 3.0];
        let shape = vec![2, 2];
        let sparse = SparseTensor::from_coo(indices, values, shape).unwrap();

        let result = sparse.mul_scalar(2.0).unwrap();
        assert_eq!(result.nnz(), 2);

        let dense_result = result.to_dense().unwrap();
        let expected = vec![4.0, 0.0, 0.0, 6.0];
        assert_eq!(dense_result.data().unwrap(), expected);
    }

    #[test]
    fn test_map_function() {
        let indices = vec![vec![0, 0], vec![1, 1]];
        let values = vec![2.0, 3.0];
        let shape = vec![2, 2];
        let sparse = SparseTensor::from_coo(indices, values, shape).unwrap();

        let result = sparse.map(|x| x * x).unwrap(); // Square all values
        assert_eq!(result.nnz(), 2);

        let dense_result = result.to_dense().unwrap();
        let expected = vec![4.0, 0.0, 0.0, 9.0];
        assert_eq!(dense_result.data().unwrap(), expected);
    }

    #[test]
    fn test_error_cases() {
        // Mismatched indices and values length
        let indices = vec![vec![0, 0]];
        let values = vec![1.0, 2.0]; // Different length
        let shape = vec![2, 2];
        assert!(SparseTensor::from_coo(indices, values, shape).is_err());

        // Out of bounds indices
        let indices = vec![vec![2, 0]]; // Row 2 is out of bounds for 2x2 matrix
        let values = vec![1.0];
        let shape = vec![2, 2];
        assert!(SparseTensor::from_coo(indices, values, shape).is_err());
    }

    // Tests for CSR format
    #[test]
    fn test_csr_creation() {
        // Create CSR matrix:
        // [[1.0, 0.0, 2.0],
        //  [0.0, 3.0, 0.0],
        //  [4.0, 0.0, 5.0]]
        let row_ptr = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 3];

        let sparse = SparseCSR::new(row_ptr, col_indices, values, shape).unwrap();
        assert_eq!(sparse.nnz(), 5);
        assert_eq!(sparse.shape(), &[3, 3]);
    }

    #[test]
    fn test_csr_to_dense() {
        let row_ptr = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 3];

        let sparse = SparseCSR::new(row_ptr, col_indices, values, shape).unwrap();
        let dense = sparse.to_dense().unwrap();

        let expected = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0];
        assert_eq!(dense.data().unwrap(), expected);
    }

    #[test]
    fn test_csr_from_coo() {
        let indices = vec![vec![0, 0], vec![0, 2], vec![1, 1], vec![2, 0], vec![2, 2]];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 3];

        let coo = SparseTensor::from_coo(indices, values, shape).unwrap();
        let csr = SparseCSR::from_coo(&coo).unwrap();

        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.shape(), &[3, 3]);

        // Verify CSR structure
        assert_eq!(csr.row_ptr(), &[0, 2, 3, 5]);
        assert_eq!(csr.col_indices(), &[0, 2, 1, 0, 2]);
    }

    #[test]
    fn test_csr_matvec() {
        // Matrix: [[1.0, 0.0], [0.0, 2.0]]
        let row_ptr = vec![0, 1, 2];
        let col_indices = vec![0, 1];
        let values = vec![1.0, 2.0];
        let shape = vec![2, 2];

        let sparse = SparseCSR::new(row_ptr, col_indices, values, shape).unwrap();
        let vec = vec![3.0, 4.0];
        let result = sparse.matvec(&vec).unwrap();

        // Expected: [1*3, 2*4] = [3.0, 8.0]
        assert_eq!(result, vec![3.0, 8.0]);
    }

    #[test]
    fn test_csr_get_row() {
        let row_ptr = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 3];

        let sparse = SparseCSR::new(row_ptr, col_indices, values, shape).unwrap();

        // Get row 0
        let (cols, vals) = sparse.get_row(0).unwrap();
        assert_eq!(cols, vec![0, 2]);
        assert_eq!(vals, vec![1.0, 2.0]);

        // Get row 1
        let (cols, vals) = sparse.get_row(1).unwrap();
        assert_eq!(cols, vec![1]);
        assert_eq!(vals, vec![3.0]);
    }

    #[test]
    fn test_csr_to_coo() {
        let row_ptr = vec![0, 2, 3, 5];
        let col_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 3];

        let csr = SparseCSR::new(row_ptr, col_indices, values, shape).unwrap();
        let coo = csr.to_coo().unwrap();

        assert_eq!(coo.nnz(), 5);
        let dense_coo = coo.to_dense().unwrap();
        let dense_csr = csr.to_dense().unwrap();
        assert_eq!(dense_coo.data().unwrap(), dense_csr.data().unwrap());
    }

    // Tests for CSC format
    #[test]
    fn test_csc_creation() {
        // Create CSC matrix:
        // [[1.0, 0.0, 2.0],
        //  [0.0, 3.0, 0.0],
        //  [4.0, 0.0, 5.0]]
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 4.0, 3.0, 2.0, 5.0];
        let shape = vec![3, 3];

        let sparse = SparseCSC::new(col_ptr, row_indices, values, shape).unwrap();
        assert_eq!(sparse.nnz(), 5);
        assert_eq!(sparse.shape(), &[3, 3]);
    }

    #[test]
    fn test_csc_to_dense() {
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 4.0, 3.0, 2.0, 5.0];
        let shape = vec![3, 3];

        let sparse = SparseCSC::new(col_ptr, row_indices, values, shape).unwrap();
        let dense = sparse.to_dense().unwrap();

        let expected = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0];
        assert_eq!(dense.data().unwrap(), expected);
    }

    #[test]
    fn test_csc_from_coo() {
        let indices = vec![vec![0, 0], vec![0, 2], vec![1, 1], vec![2, 0], vec![2, 2]];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 3];

        let coo = SparseTensor::from_coo(indices, values, shape).unwrap();
        let csc = SparseCSC::from_coo(&coo).unwrap();

        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.shape(), &[3, 3]);

        // Verify CSC structure
        assert_eq!(csc.col_ptr(), &[0, 2, 3, 5]);
        assert_eq!(csc.row_indices(), &[0, 2, 1, 0, 2]);
    }

    #[test]
    fn test_csc_transpose_matvec() {
        // Matrix: [[1.0, 0.0], [0.0, 2.0]]
        // CSC stores column-wise
        let col_ptr = vec![0, 1, 2];
        let row_indices = vec![0, 1];
        let values = vec![1.0, 2.0];
        let shape = vec![2, 2];

        let sparse = SparseCSC::new(col_ptr, row_indices, values, shape).unwrap();
        let vec = vec![3.0, 4.0];
        let result = sparse.transpose_matvec(&vec).unwrap();

        // Expected: A^T * [3, 4] = [[1, 0], [0, 2]]^T * [3, 4] = [3.0, 8.0]
        assert_eq!(result, vec![3.0, 8.0]);
    }

    #[test]
    fn test_csc_get_col() {
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 4.0, 3.0, 2.0, 5.0];
        let shape = vec![3, 3];

        let sparse = SparseCSC::new(col_ptr, row_indices, values, shape).unwrap();

        // Get column 0
        let (rows, vals) = sparse.get_col(0).unwrap();
        assert_eq!(rows, vec![0, 2]);
        assert_eq!(vals, vec![1.0, 4.0]);

        // Get column 1
        let (rows, vals) = sparse.get_col(1).unwrap();
        assert_eq!(rows, vec![1]);
        assert_eq!(vals, vec![3.0]);
    }

    #[test]
    fn test_csc_to_coo() {
        let col_ptr = vec![0, 2, 3, 5];
        let row_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 4.0, 3.0, 2.0, 5.0];
        let shape = vec![3, 3];

        let csc = SparseCSC::new(col_ptr, row_indices, values, shape).unwrap();
        let coo = csc.to_coo().unwrap();

        assert_eq!(coo.nnz(), 5);
        let dense_coo = coo.to_dense().unwrap();
        let dense_csc = csc.to_dense().unwrap();
        assert_eq!(dense_coo.data().unwrap(), dense_csc.data().unwrap());
    }

    #[test]
    fn test_format_conversions() {
        // Test round-trip: COO -> CSR -> COO -> CSC -> COO
        let indices = vec![vec![0, 0], vec![0, 2], vec![1, 1], vec![2, 0], vec![2, 2]];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![3, 3];

        let coo1 = SparseTensor::from_coo(indices, values, shape).unwrap();
        let csr = SparseCSR::from_coo(&coo1).unwrap();
        let coo2 = csr.to_coo().unwrap();
        let csc = SparseCSC::from_coo(&coo2).unwrap();
        let coo3 = csc.to_coo().unwrap();

        // All should represent the same matrix
        let dense1 = coo1.to_dense().unwrap();
        let dense2 = coo2.to_dense().unwrap();
        let dense3 = coo3.to_dense().unwrap();

        assert_eq!(dense1.data().unwrap(), dense2.data().unwrap());
        assert_eq!(dense2.data().unwrap(), dense3.data().unwrap());
    }

    #[test]
    fn test_csr_error_cases() {
        // Wrong shape length
        let row_ptr = vec![0, 1];
        let col_indices = vec![0];
        let values = vec![1.0];
        let shape = vec![1]; // 1D not supported
        assert!(SparseCSR::new(row_ptr, col_indices, values, shape).is_err());

        // Mismatched lengths
        let row_ptr = vec![0, 2];
        let col_indices = vec![0];
        let values = vec![1.0, 2.0]; // Different from col_indices
        let shape = vec![1, 2];
        assert!(SparseCSR::new(row_ptr, col_indices, values, shape).is_err());

        // Non-monotonic row pointers
        let row_ptr = vec![0, 2, 1]; // Decreases
        let col_indices = vec![0, 1];
        let values = vec![1.0, 2.0];
        let shape = vec![2, 2];
        assert!(SparseCSR::new(row_ptr, col_indices, values, shape).is_err());

        // Column index out of bounds
        let row_ptr = vec![0, 1];
        let col_indices = vec![5]; // Out of bounds
        let values = vec![1.0];
        let shape = vec![1, 2];
        assert!(SparseCSR::new(row_ptr, col_indices, values, shape).is_err());
    }

    #[test]
    fn test_csc_error_cases() {
        // Wrong shape length
        let col_ptr = vec![0, 1];
        let row_indices = vec![0];
        let values = vec![1.0];
        let shape = vec![1]; // 1D not supported
        assert!(SparseCSC::new(col_ptr, row_indices, values, shape).is_err());

        // Mismatched lengths
        let col_ptr = vec![0, 2];
        let row_indices = vec![0];
        let values = vec![1.0, 2.0]; // Different from row_indices
        let shape = vec![2, 1];
        assert!(SparseCSC::new(col_ptr, row_indices, values, shape).is_err());

        // Non-monotonic column pointers
        let col_ptr = vec![0, 2, 1]; // Decreases
        let row_indices = vec![0, 1];
        let values = vec![1.0, 2.0];
        let shape = vec![2, 2];
        assert!(SparseCSC::new(col_ptr, row_indices, values, shape).is_err());

        // Row index out of bounds
        let col_ptr = vec![0, 1];
        let row_indices = vec![5]; // Out of bounds
        let values = vec![1.0];
        let shape = vec![2, 1];
        assert!(SparseCSC::new(col_ptr, row_indices, values, shape).is_err());
    }
}
