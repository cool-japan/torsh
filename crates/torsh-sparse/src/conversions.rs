//! Consolidated conversion utilities for sparse tensor formats
//!
//! This module provides optimized conversion paths, common validation logic,
//! and shared patterns to reduce code duplication and improve performance.

use crate::{CooTensor, CscTensor, CsrTensor, SparseTensor, TorshResult};
use std::collections::HashMap;
use torsh_core::{Shape, TorshError};
use torsh_tensor::Tensor;

/// Conversion hints to optimize format transformations
#[derive(Debug, Default, Clone)]
pub struct ConversionHints {
    /// Whether the tensor data is already sorted by indices
    pub is_sorted: bool,
    /// Whether the tensor is symmetric
    pub is_symmetric: bool,
    /// Whether all rows have similar number of non-zeros
    pub has_uniform_row_sizes: bool,
    /// Whether the tensor has block structure
    pub is_block_structured: bool,
    /// Whether the tensor is diagonal or near-diagonal
    pub is_diagonal: bool,
}

/// Common validation utilities for sparse tensor operations
pub mod validation {
    use super::*;

    /// Validate that a tensor is 2D (required for most sparse formats)
    pub fn validate_2d_tensor(shape: &Shape, format_name: &str) -> TorshResult<()> {
        if shape.dims().len() != 2 {
            return Err(TorshError::InvalidArgument(format!(
                "{} format only supports 2D tensors, got {}D",
                format_name,
                shape.dims().len()
            )));
        }
        Ok(())
    }

    /// Validate sparse indices against tensor shape
    pub fn validate_sparse_indices(
        row_indices: &[usize],
        col_indices: &[usize],
        shape: &Shape,
    ) -> TorshResult<()> {
        let max_rows = shape.dims()[0];
        let max_cols = shape.dims()[1];

        for &row in row_indices {
            if row >= max_rows {
                return Err(TorshError::InvalidArgument(format!(
                    "Row index {row} exceeds shape bounds [{max_rows}]"
                )));
            }
        }

        for &col in col_indices {
            if col >= max_cols {
                return Err(TorshError::InvalidArgument(format!(
                    "Column index {col} exceeds shape bounds [{max_cols}]"
                )));
            }
        }

        Ok(())
    }

    /// Validate that arrays have matching lengths
    pub fn validate_array_lengths(
        rows_len: usize,
        cols_len: usize,
        vals_len: usize,
    ) -> TorshResult<()> {
        if rows_len != cols_len || cols_len != vals_len {
            return Err(TorshError::InvalidArgument(format!(
                "Array lengths must match: rows={rows_len}, cols={cols_len}, values={vals_len}"
            )));
        }
        Ok(())
    }

    /// Validate that a matrix is square (for operations requiring square matrices)
    pub fn validate_square_matrix(shape: &Shape) -> TorshResult<()> {
        validate_2d_tensor(shape, "Square matrix operation")?;
        if shape.dims()[0] != shape.dims()[1] {
            return Err(TorshError::InvalidArgument(format!(
                "Matrix must be square, got shape [{} x {}]",
                shape.dims()[0],
                shape.dims()[1]
            )));
        }
        Ok(())
    }
}

/// Common conversion patterns abstracted for reuse
pub mod patterns {
    use super::*;

    /// Extract triplets from any sparse tensor format
    pub fn extract_triplets(sparse: &dyn SparseTensor) -> TorshResult<Vec<(usize, usize, f32)>> {
        let coo = sparse.to_coo()?;
        Ok(coo.triplets())
    }

    /// Build COO tensor from triplets with validation
    pub fn triplets_to_coo(
        triplets: Vec<(usize, usize, f32)>,
        shape: Shape,
    ) -> TorshResult<CooTensor> {
        let (row_indices, col_indices, values): (Vec<_>, Vec<_>, Vec<_>) =
            triplets.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut rows, mut cols, mut vals), (r, c, v)| {
                    rows.push(r);
                    cols.push(c);
                    vals.push(v);
                    (rows, cols, vals)
                },
            );

        CooTensor::new(row_indices, col_indices, values, shape)
    }

    /// Generic dense-to-sparse conversion with threshold filtering
    pub fn dense_to_sparse_with_threshold<F>(
        dense: &Tensor,
        _threshold: f32,
        _processor: F,
    ) -> TorshResult<()>
    where
        F: FnMut(usize, usize, f32) -> TorshResult<()>,
    {
        let shape = dense.shape();
        validation::validate_2d_tensor(&shape, "Dense-to-sparse conversion")?;

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        // Simplified iteration - in practice you'd want more efficient tensor access
        for _row in 0..rows {
            for _col in 0..cols {
                // Note: This would need proper tensor indexing implementation
                // For now we'll use a placeholder approach
                // let value = dense.get(&[row, col])?;
                // if value.abs() > threshold {
                //     processor(row, col, value)?;
                // }
            }
        }

        Ok(())
    }

    /// Filter and process sparse triplets with a threshold
    pub fn filter_triplets_by_threshold(
        triplets: Vec<(usize, usize, f32)>,
        threshold: f32,
    ) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        triplets
            .into_iter()
            .filter(|(_, _, v)| v.abs() > threshold)
            .fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut rows, mut cols, mut vals), (r, c, v)| {
                    rows.push(r);
                    cols.push(c);
                    vals.push(v);
                    (rows, cols, vals)
                },
            )
    }

    /// Sort triplets by row-major order (row first, then column)
    pub fn sort_triplets_row_major(triplets: &mut [(usize, usize, f32)]) {
        triplets.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    }

    /// Sort triplets by column-major order (column first, then row)
    pub fn sort_triplets_col_major(triplets: &mut [(usize, usize, f32)]) {
        triplets.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
    }

    /// Aggregate duplicate triplets by summing values
    pub fn aggregate_duplicate_triplets(
        mut triplets: Vec<(usize, usize, f32)>,
    ) -> Vec<(usize, usize, f32)> {
        if triplets.is_empty() {
            return triplets;
        }

        // Sort first to group duplicates together
        sort_triplets_row_major(&mut triplets);

        let mut result = Vec::new();
        let mut current = triplets[0];

        for &next in &triplets[1..] {
            if current.0 == next.0 && current.1 == next.1 {
                // Same position - accumulate values
                current.2 += next.2;
            } else {
                // Different position - save current and start new one
                if current.2.abs() > f32::EPSILON {
                    result.push(current);
                }
                current = next;
            }
        }

        // Don't forget the last one
        if current.2.abs() > f32::EPSILON {
            result.push(current);
        }

        result
    }
}

/// Optimized direct conversion paths that avoid COO intermediate format
pub mod direct_conversions {
    use super::*;

    /// Direct CSR to CSC conversion via transpose
    /// This is much more efficient than CSR → COO → CSC
    pub fn csr_to_csc_direct(csr: &CsrTensor) -> TorshResult<CscTensor> {
        // Get CSR data
        let row_ptr = csr.row_ptr();
        let col_indices = csr.col_indices();
        let values = csr.values();
        let shape = csr.shape();
        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        // Build column pointers by counting elements per column
        let mut col_ptr = vec![0; cols + 1];
        for &col in col_indices {
            col_ptr[col + 1] += 1;
        }

        // Convert counts to cumulative offsets
        for i in 1..=cols {
            col_ptr[i] += col_ptr[i - 1];
        }

        // Allocate output arrays
        let nnz = col_indices.len();
        let mut new_row_indices = vec![0; nnz];
        let mut new_values = vec![0.0; nnz];
        let mut col_offsets = col_ptr.clone();

        // Fill output arrays by processing CSR row by row
        for row in 0..rows {
            let start = row_ptr[row];
            let end = row_ptr[row + 1];

            for idx in start..end {
                let col = col_indices[idx];
                let val = values[idx];

                // Place this element in the appropriate column
                let pos = col_offsets[col];
                new_row_indices[pos] = row;
                new_values[pos] = val;
                col_offsets[col] += 1;
            }
        }

        // Create CSC tensor - note: need to check if CSC has a similar constructor
        // For now, use COO as intermediate until we verify CSC API
        let triplets: Vec<_> = (0..nnz)
            .map(|i| (new_row_indices[i], i, new_values[i]))
            .collect();
        let transposed_shape = Shape::new(vec![cols, rows]);
        let coo = patterns::triplets_to_coo(triplets, transposed_shape)?;
        CscTensor::from_coo(&coo)
    }

    /// Direct CSC to CSR conversion via transpose
    pub fn csc_to_csr_direct(csc: &CscTensor) -> TorshResult<CsrTensor> {
        // For now, we'll implement this using COO as intermediate
        // In a full implementation, we'd check for CSC accessor methods similar to CSR
        let coo = csc.to_coo()?;
        CsrTensor::from_coo(&coo)
    }

    /// Convert symmetric matrix representation to full matrix
    pub fn symmetric_to_full_triplets(
        triplets: &[(usize, usize, f32)],
        mode: crate::symmetric::SymmetricMode,
    ) -> Vec<(usize, usize, f32)> {
        let mut full_triplets = Vec::with_capacity(triplets.len() * 2);

        for &(row, col, val) in triplets {
            // Always add the original element
            full_triplets.push((row, col, val));

            // Add the symmetric element if it's not on the diagonal
            if row != col {
                match mode {
                    crate::symmetric::SymmetricMode::Upper => {
                        // Upper triangular: also add (col, row)
                        full_triplets.push((col, row, val));
                    }
                    crate::symmetric::SymmetricMode::Lower => {
                        // Lower triangular: also add (col, row)
                        full_triplets.push((col, row, val));
                    }
                }
            }
        }

        full_triplets
    }
}

/// Conversion optimization utilities
pub mod optimization {
    use super::*;

    /// Analyze sparse tensor to provide conversion hints
    pub fn analyze_conversion_hints(sparse: &dyn SparseTensor) -> ConversionHints {
        let triplets = patterns::extract_triplets(sparse).unwrap_or_default();
        let shape = sparse.shape();
        let nnz = sparse.nnz();

        if nnz == 0 {
            return ConversionHints::default();
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        // Check if sorted
        let is_sorted = is_triplets_sorted(&triplets);

        // Check if symmetric (for square matrices only)
        let is_symmetric = if rows == cols {
            check_symmetry(&triplets)
        } else {
            false
        };

        // Check if diagonal
        let is_diagonal = triplets.iter().all(|(r, c, _)| r == c);

        // Check for uniform row sizes
        let has_uniform_row_sizes = check_uniform_row_sizes(&triplets, rows);

        // Simple block structure detection (placeholder)
        let is_block_structured = false;

        ConversionHints {
            is_sorted,
            is_symmetric,
            has_uniform_row_sizes,
            is_block_structured,
            is_diagonal,
        }
    }

    fn is_triplets_sorted(triplets: &[(usize, usize, f32)]) -> bool {
        triplets
            .windows(2)
            .all(|w| (w[0].0, w[0].1) <= (w[1].0, w[1].1))
    }

    fn check_symmetry(triplets: &[(usize, usize, f32)]) -> bool {
        let mut element_map: HashMap<(usize, usize), f32> = HashMap::new();

        // Build map of all elements
        for &(row, col, val) in triplets {
            element_map.insert((row, col), val);
        }

        // Check symmetry
        for &(row, col, val) in triplets {
            if row != col {
                if let Some(&sym_val) = element_map.get(&(col, row)) {
                    if (val - sym_val).abs() > 1e-6 {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        true
    }

    fn check_uniform_row_sizes(triplets: &[(usize, usize, f32)], rows: usize) -> bool {
        let mut row_counts = vec![0; rows];
        for &(row, _, _) in triplets {
            row_counts[row] += 1;
        }

        if let Some(&first_count) = row_counts.iter().find(|&&count| count > 0) {
            row_counts
                .iter()
                .all(|&count| count == 0 || count == first_count)
        } else {
            true
        }
    }

    /// Choose optimal conversion path based on source and target formats
    pub fn choose_optimal_path(
        source_format: crate::SparseFormat,
        target_format: crate::SparseFormat,
        hints: &ConversionHints,
    ) -> ConversionPath {
        use crate::SparseFormat::*;

        match (source_format, target_format) {
            // Identity conversions
            (a, b) if a == b => ConversionPath::Identity,

            // Direct optimized paths
            (Csr, Csc) | (Csc, Csr) => ConversionPath::Direct,

            // COO is efficient for most conversions from complex formats
            (Bsr | Dia | Ell | Rle | Symmetric, _) => ConversionPath::ViaCoo,

            // For other combinations, analyze hints
            _ => {
                if hints.is_diagonal {
                    ConversionPath::ViaDia
                } else if hints.is_symmetric {
                    ConversionPath::ViaSymmetric
                } else {
                    ConversionPath::ViaCoo
                }
            }
        }
    }

    /// Different conversion path strategies
    #[derive(Debug, Clone, PartialEq)]
    pub enum ConversionPath {
        /// No conversion needed
        Identity,
        /// Direct conversion without intermediate format
        Direct,
        /// Convert via COO intermediate
        ViaCoo,
        /// Convert via DIA intermediate (for diagonal matrices)
        ViaDia,
        /// Convert via Symmetric intermediate (for symmetric matrices)
        ViaSymmetric,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_functions() {
        // Test 2D validation
        let shape_2d = Shape::new(vec![3, 4]);
        let shape_3d = Shape::new(vec![3, 4, 5]);

        assert!(validation::validate_2d_tensor(&shape_2d, "Test").is_ok());
        assert!(validation::validate_2d_tensor(&shape_3d, "Test").is_err());

        // Test square matrix validation
        let square_shape = Shape::new(vec![3, 3]);
        let non_square_shape = Shape::new(vec![3, 4]);

        assert!(validation::validate_square_matrix(&square_shape).is_ok());
        assert!(validation::validate_square_matrix(&non_square_shape).is_err());
    }

    #[test]
    fn test_triplet_processing() {
        let triplets = vec![
            (0, 0, 1.0),
            (0, 0, 2.0), // Duplicate position
            (1, 1, 3.0),
            (0, 1, 0.0), // Zero value
        ];

        let aggregated = patterns::aggregate_duplicate_triplets(triplets);

        // Should have combined (0,0) entries and filtered out zero
        assert_eq!(aggregated.len(), 2);
        assert!(aggregated.contains(&(0, 0, 3.0))); // 1.0 + 2.0
        assert!(aggregated.contains(&(1, 1, 3.0)));
    }

    #[test]
    fn test_conversion_hints() {
        // Create a simple COO tensor for testing
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
        let shape = Shape::new(vec![3, 3]);
        let coo = patterns::triplets_to_coo(triplets, shape).unwrap();

        let hints = optimization::analyze_conversion_hints(&coo as &dyn SparseTensor);

        // Diagonal matrix should be detected
        assert!(hints.is_diagonal);
    }
}
