//! SciRS2 integration and core arithmetic for sparse COO tensors
//!
//! This module provides:
//!
//! - Native sparse addition over the [`SparseTensor`] trait, producing a
//!   coalesced [`CooTensor`].
//! - Coalescing / canonicalization of sparse tensors (merging duplicate
//!   coordinates and dropping numerical zeros).
//! - Optional zero-copy-style conversions to and from the `scirs2-sparse`
//!   [`scirs2_sparse::CooArray`] type, available with the
//!   `scirs2-integration` feature.

use crate::{CooTensor, SparseTensor, TorshResult};
use std::collections::BTreeMap;
use torsh_core::TorshError;

/// Numerical threshold below which an accumulated value is treated as a
/// structural zero and dropped from the sparse result.
const ZERO_THRESHOLD: f32 = 1e-12;

/// Perform sparse matrix addition, returning a coalesced COO tensor.
///
/// Both operands are first converted to COO form (via the [`SparseTensor`]
/// trait), then their `(row, col, value)` triplets are merged: entries that
/// share the same coordinate are summed, and any coordinate whose accumulated
/// value falls below [`ZERO_THRESHOLD`] is dropped. The result is returned in
/// canonical (row-major sorted, duplicate-free) order.
///
/// # Errors
/// Returns [`TorshError::InvalidArgument`] if the two operands do not have
/// matching shapes.
pub fn scirs2_add(a: &dyn SparseTensor, b: &dyn SparseTensor) -> TorshResult<CooTensor> {
    let a_coo = a.to_coo()?;
    let b_coo = b.to_coo()?;

    if a_coo.shape() != b_coo.shape() {
        return Err(TorshError::InvalidArgument(format!(
            "Sparse addition requires matching shapes, got {:?} and {:?}",
            a_coo.shape().dims(),
            b_coo.shape().dims()
        )));
    }

    let shape = a_coo.shape().clone();
    let cols = shape.dims()[1];

    // Accumulate into a sorted map keyed by the flattened (row, col) index so
    // that the output is naturally produced in canonical row-major order.
    let mut accum: BTreeMap<usize, f32> = BTreeMap::new();

    for (row, col, value) in a_coo.triplets() {
        *accum.entry(row * cols + col).or_insert(0.0) += value;
    }
    for (row, col, value) in b_coo.triplets() {
        *accum.entry(row * cols + col).or_insert(0.0) += value;
    }

    let mut rows = Vec::with_capacity(accum.len());
    let mut col_indices = Vec::with_capacity(accum.len());
    let mut values = Vec::with_capacity(accum.len());

    for (flat, value) in accum {
        if value.abs() < ZERO_THRESHOLD {
            continue;
        }
        rows.push(flat / cols);
        col_indices.push(flat % cols);
        values.push(value);
    }

    CooTensor::new(rows, col_indices, values, shape)
}

/// Coalesce a sparse tensor into canonical COO form.
///
/// Converts the input to COO, merges any duplicate coordinates by summing
/// their values, drops explicit zeros, and returns the entries in row-major
/// sorted order. This is the canonical normal form expected by most
/// downstream sparse algorithms.
pub fn scirs2_enhanced_ops(tensor: &dyn SparseTensor) -> TorshResult<CooTensor> {
    let coo = tensor.to_coo()?;
    let shape = coo.shape().clone();
    let cols = shape.dims()[1];

    let mut accum: BTreeMap<usize, f32> = BTreeMap::new();
    for (row, col, value) in coo.triplets() {
        *accum.entry(row * cols + col).or_insert(0.0) += value;
    }

    let mut rows = Vec::with_capacity(accum.len());
    let mut col_indices = Vec::with_capacity(accum.len());
    let mut values = Vec::with_capacity(accum.len());

    for (flat, value) in accum {
        if value.abs() < ZERO_THRESHOLD {
            continue;
        }
        rows.push(flat / cols);
        col_indices.push(flat % cols);
        values.push(value);
    }

    CooTensor::new(rows, col_indices, values, shape)
}

/// Convert a torsh COO tensor to a `scirs2-sparse` COO array.
#[cfg(feature = "scirs2-integration")]
pub fn coo_to_scirs2(tensor: &CooTensor) -> TorshResult<scirs2_sparse::CooArray<f32>> {
    use scirs2_sparse::SparseArray;

    let triplets = tensor.triplets();
    let mut rows = Vec::with_capacity(triplets.len());
    let mut cols = Vec::with_capacity(triplets.len());
    let mut data = Vec::with_capacity(triplets.len());
    for (r, c, v) in triplets {
        rows.push(r);
        cols.push(c);
        data.push(v);
    }

    let dims = tensor.shape().dims();
    let shape = (dims[0], dims[1]);
    let array = scirs2_sparse::CooArray::from_triplets(&rows, &cols, &data, shape, false)
        .map_err(|e| TorshError::InvalidArgument(format!("scirs2 error: {e:?}")))?;
    // Touch the trait method so the import is exercised and the array shape is validated.
    debug_assert_eq!(array.shape(), shape);
    Ok(array)
}

/// Convert a `scirs2-sparse` COO array back to a torsh COO tensor.
#[cfg(feature = "scirs2-integration")]
pub fn scirs2_to_coo(array: &scirs2_sparse::CooArray<f32>) -> TorshResult<CooTensor> {
    use scirs2_sparse::SparseArray;

    let rows = array.get_rows().to_vec();
    let cols = array.get_cols().to_vec();
    let data = array.get_data().to_vec();
    let (r, c) = array.shape();
    let shape = torsh_core::Shape::new(vec![r, c]);

    CooTensor::new(rows, cols, data, shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn test_add_disjoint_patterns() {
        // a and b have non-overlapping coordinates: result should contain both.
        let a = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![2.0, 3.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();
        let b = CooTensor::new(
            vec![0, 1],
            vec![1, 0],
            vec![1.0, 4.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();

        let result = scirs2_add(&a as &dyn SparseTensor, &b as &dyn SparseTensor).unwrap();
        let dense = result.to_dense().unwrap().to_vec().unwrap();
        // [[2, 1], [4, 3]]
        assert_eq!(dense, vec![2.0, 1.0, 4.0, 3.0]);
        assert_eq!(result.nnz(), 4);
    }

    #[test]
    fn test_add_overlapping_coords_sum() {
        // Overlapping coordinates must be summed, not duplicated.
        let a = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![2.0, 3.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();
        let b = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![5.0, 7.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();

        let result = scirs2_add(&a as &dyn SparseTensor, &b as &dyn SparseTensor).unwrap();
        let dense = result.to_dense().unwrap().to_vec().unwrap();
        // diagonal: 2+5=7, 3+7=10
        assert_eq!(dense, vec![7.0, 0.0, 0.0, 10.0]);
        assert_eq!(result.nnz(), 2);
    }

    #[test]
    fn test_add_zero_tensor_preserves_values() {
        // Adding an all-zero sparse tensor must preserve the original values.
        let a = CooTensor::new(
            vec![0, 1, 1],
            vec![0, 0, 1],
            vec![1.5, -2.5, 4.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();
        let zero = CooTensor::new(vec![], vec![], vec![], Shape::new(vec![2, 2])).unwrap();

        let result = scirs2_add(&a as &dyn SparseTensor, &zero as &dyn SparseTensor).unwrap();
        let dense = result.to_dense().unwrap().to_vec().unwrap();
        assert_eq!(dense, vec![1.5, 0.0, -2.5, 4.0]);
    }

    #[test]
    fn test_add_cancellation_drops_zero() {
        // a + (-a) must produce a structurally empty result.
        let a = CooTensor::new(vec![0], vec![1], vec![3.0], Shape::new(vec![2, 2])).unwrap();
        let neg = CooTensor::new(vec![0], vec![1], vec![-3.0], Shape::new(vec![2, 2])).unwrap();

        let result = scirs2_add(&a as &dyn SparseTensor, &neg as &dyn SparseTensor).unwrap();
        assert_eq!(result.nnz(), 0);
    }

    #[test]
    fn test_add_shape_mismatch_errors() {
        let a = CooTensor::new(vec![0], vec![0], vec![1.0], Shape::new(vec![2, 2])).unwrap();
        let b = CooTensor::new(vec![0], vec![0], vec![1.0], Shape::new(vec![3, 3])).unwrap();
        assert!(scirs2_add(&a as &dyn SparseTensor, &b as &dyn SparseTensor).is_err());
    }

    #[test]
    fn test_enhanced_ops_coalesces() {
        // Build a COO with a deliberately unsorted, value-only layout and verify
        // canonicalization produces sorted, deduplicated output.
        let t = CooTensor::new(
            vec![1, 0, 1],
            vec![1, 0, 0],
            vec![3.0, 1.0, 2.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();
        let result = scirs2_enhanced_ops(&t as &dyn SparseTensor).unwrap();
        let triplets = result.triplets();
        // Canonical row-major order: (0,0), (1,0), (1,1)
        assert_eq!(triplets, vec![(0, 0, 1.0), (1, 0, 2.0), (1, 1, 3.0)]);
    }

    #[cfg(feature = "scirs2-integration")]
    #[test]
    fn test_scirs2_roundtrip() {
        let coo = CooTensor::new(
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![1.0, 2.0, 3.0],
            Shape::new(vec![3, 3]),
        )
        .unwrap();

        let scirs2_coo = coo_to_scirs2(&coo).unwrap();
        let coo_back = scirs2_to_coo(&scirs2_coo).unwrap();
        assert_eq!(coo_back.nnz(), 3);
        let dense = coo_back.to_dense().unwrap().to_vec().unwrap();
        assert_eq!(dense, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }
}
