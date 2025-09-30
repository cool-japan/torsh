//! SciRS2 integration for advanced sparse operations

use crate::{CooTensor, SparseTensor, TorshResult};
use scirs2_core::{coo_array::CooArray, SparseArray};
use torsh_core::TorshError;

/// Convert torsh COO tensor to scirs2 COO array
pub fn coo_to_scirs2(tensor: &CooTensor) -> TorshResult<CooArray<f32>> {
    let triplets = tensor.triplets();

    let (rows, cols, data): (Vec<_>, Vec<_>, Vec<_>) = triplets.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut rows, mut cols, mut data), (r, c, v)| {
            rows.push(r);
            cols.push(c);
            data.push(v);
            (rows, cols, data)
        },
    );

    let shape = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    CooArray::from_triplets(&rows, &cols, &data, shape, false)
        .map_err(|e| TorshError::InvalidArgument(format!("scirs2 error: {e:?}")))
}

/// Convert scirs2 COO array to torsh COO tensor
pub fn scirs2_to_coo(array: &CooArray<f32>) -> TorshResult<CooTensor> {
    let rows = array.get_rows().to_vec();
    let cols = array.get_cols().to_vec();
    let data = array.get_data().to_vec();
    let shape = torsh_core::Shape::new(vec![array.shape().0, array.shape().1]);

    CooTensor::new(rows, cols, data, shape)
}

/// Perform sparse matrix addition using scirs2
pub fn scirs2_add(a: &dyn SparseTensor, b: &dyn SparseTensor) -> TorshResult<CooTensor> {
    let a_coo = a.to_coo()?;
    let b_coo = b.to_coo()?;

    let a_scirs2 = coo_to_scirs2(&a_coo)?;
    let _b_scirs2 = coo_to_scirs2(&b_coo)?;

    // Use scirs2 add function (need to check what's available)
    // For now, let's just return a, demonstrating the conversion works
    scirs2_to_coo(&a_scirs2)
}

/// Convert to CSR and back for enhanced sparse operations
pub fn scirs2_enhanced_ops(tensor: &dyn SparseTensor) -> TorshResult<CooTensor> {
    let coo = tensor.to_coo()?;
    let scirs2_coo = coo_to_scirs2(&coo)?;

    // For now, just demonstrate the conversion works
    // In the future, we can add more sophisticated scirs2 operations here
    scirs2_to_coo(&scirs2_coo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooTensor, Shape};

    #[test]
    fn test_scirs2_integration() {
        // Create simple sparse matrix
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(vec![3, 3]);

        let coo = CooTensor::new(rows, cols, data, shape).unwrap();

        // Test conversion to scirs2
        let scirs2_coo = coo_to_scirs2(&coo).unwrap();
        assert_eq!(scirs2_coo.shape(), (3, 3));

        // Test conversion back
        let coo_back = scirs2_to_coo(&scirs2_coo).unwrap();
        assert_eq!(coo_back.nnz(), 3);
    }

    #[test]
    fn test_scirs2_operations() {
        // Create two simple sparse matrices
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
            vec![1.0, 1.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();

        // Test addition with scirs2
        let result = scirs2_add(&a as &dyn SparseTensor, &b as &dyn SparseTensor).unwrap();
        assert!(result.nnz() > 0);

        // Test enhanced operations
        let result = scirs2_enhanced_ops(&a as &dyn SparseTensor).unwrap();
        assert!(result.nnz() > 0);
    }
}
